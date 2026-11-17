#!/usr/bin/env python3
"""合成主机侧负载：不需要加载模型，直接复现 DeepSeek 推理时那三类 host 流量。

  --dram GB/s   DRAM 读带宽：N 线程对大块 CPU 张量反复求和
  --nvme GB/s   NVMe 读带宽：O_DIRECT 随机 1MiB 块（绕开页缓存）
  --h2d  GB/s   pinned host→device 拷贝：在多张卡上轮转
  --sm          额外让 4 张卡跑 matmul，保持 SM 忙

每 2 秒打印实测速率，用来证明"负载真的在跑"，而不是只有进程存在。
用法:
  python3 synthetic_load.py --dram 44 --nvme 2 --h2d 1.5 --threads 16 --duration 180
"""
import argparse
import mmap
import os
import random
import threading
import time

import torch

STOP_AT = 0.0


def now() -> float:
    return time.time()


class Meter:
    def __init__(self) -> None:
        self.bytes = 0
        self.lock = threading.Lock()

    def add(self, n: int) -> None:
        with self.lock:
            self.bytes += n


def pacer(target_bps: float, nbytes: int, t0: float) -> None:
    """按目标带宽配速，避免把机器压成另一种形态。"""
    if target_bps <= 0:
        return
    wait = nbytes / target_bps - (now() - t0)
    if wait > 0:
        time.sleep(wait)


def dram_worker(buf, meter: Meter, target_bps: float, threads: int, target_ref=None) -> None:
    torch.set_num_threads(threads)
    f32 = buf.view(torch.float32)
    nbytes = f32.numel() * 4
    while now() < STOP_AT:
        t0 = now()
        torch.sum(f32)
        meter.add(nbytes)
        cur = target_ref["dram"] if target_ref else target_bps
        pacer(cur, nbytes, t0)


def nvme_open(path: str):
    """优先 O_DIRECT（真读盘），失败则退回带 DONTNEED 的普通读。"""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        return fd, True
    except OSError as e:
        print(f"[nvme] O_DIRECT 不可用（{e}），退回普通读", flush=True)
        return os.open(path, os.O_RDONLY), False


def nvme_worker(path: str, meter: Meter, target_bps: float, chunk: int, seed: int) -> None:
    fd, direct = nvme_open(path)
    size = os.fstat(fd).st_size
    buf = mmap.mmap(-1, chunk)  # 页对齐，满足 O_DIRECT 要求
    rnd = random.Random(seed)
    while now() < STOP_AT:
        off = rnd.randrange(0, max(1, size - chunk)) & ~4095
        t0 = now()
        try:
            n = os.preadv(fd, [buf], off)
        except OSError as e:
            print(f"[nvme] pread 失败 {e}", flush=True)
            break
        if not direct:
            try:
                os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
            except Exception:
                pass
        meter.add(n)
        pacer(target_bps, n, t0)
    os.close(fd)


def h2d_worker(gpus, meter: Meter, target_bps: float, chunk_mb: int) -> None:
    src = torch.empty(chunk_mb * 2**20, dtype=torch.uint8, pin_memory=True)
    dsts = [torch.empty(src.numel(), dtype=torch.uint8, device=f"cuda:{g}") for g in gpus]
    i = 0
    while now() < STOP_AT:
        dst = dsts[i % len(dsts)]
        i += 1
        t0 = now()
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(dst.device)
        meter.add(src.numel())
        pacer(target_bps, src.numel(), t0)


def stage_worker(path: str, meter: Meter, target_bps: float, chunk: int,
                 gpu: int, churn_bytes: int, reserve_bytes: int = 0) -> None:
    """模拟 ComfyUI DynamicVRAM 的搬运：文件→pinned→显存，并制造 cudaMalloc 抖动。"""
    fd = os.open(path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    resident = None
    if reserve_bytes > 0:                       # 模拟"模型已常驻显存"的占用
        resident = torch.empty(reserve_bytes, dtype=torch.uint8, device=f"cuda:{gpu}")
        resident.fill_(1)
        print(f"[stage] 显存常驻 {reserve_bytes / 2**30:.1f} GiB 已在 cuda:{gpu}", flush=True)
    pinned = torch.empty(chunk, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(chunk, dtype=torch.uint8, device=f"cuda:{gpu}")
    churn = None
    if churn_bytes > 0:
        churn = torch.empty(churn_bytes, dtype=torch.uint8, device=f"cuda:{gpu}")
    n_cycle = 0
    while now() < STOP_AT:
        off = (n_cycle * chunk) % max(1, size - chunk)
        n_cycle += 1
        t0 = now()
        view = torch.frombuffer(mm, dtype=torch.uint8, count=chunk, offset=off)
        pinned.copy_(view)                       # 文件/页缓存 -> pinned（CPU memcpy）
        dst.copy_(pinned, non_blocking=True)     # pinned -> 显存（DMA）
        torch.cuda.synchronize(gpu)
        meter.add(chunk * 2)                     # 计两份：读 + 写
        if churn is not None and n_cycle % 4 == 0:
            churn.fill_(1)                       # 逼它真的落显存
            churn = torch.empty(churn_bytes, dtype=torch.uint8, device=f"cuda:{gpu}")
        pacer(target_bps, chunk * 2, t0)
    del pinned, dst, churn
    mm.close()
    os.close(fd)



def sm_worker(gpus, size: int) -> None:
    mats = []
    for g in gpus:
        x = torch.randn(size, size, device=f"cuda:{g}", dtype=torch.bfloat16)
        mats.append((g, x))
    while now() < STOP_AT:
        for g, x in mats:
            y = x @ x
            torch.cuda.synchronize(g)
            del y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dram", type=float, default=0.0, help="DRAM 读目标 GB/s")
    ap.add_argument("--dram-steps", default="",
                    help="阶梯爬升，如 '10,15,20,25,30,35,40,44'（配合 --step-secs）")
    ap.add_argument("--step-secs", type=float, default=22.0, help="每级持续秒数")
    ap.add_argument("--nvme", type=float, default=0.0, help="NVMe 读目标 GB/s")
    ap.add_argument("--h2d", type=float, default=0.0, help="host→device 目标 GB/s")
    ap.add_argument("--dram-gb", type=float, default=16.0, help="DRAM 工作集大小 GiB")
    ap.add_argument("--threads", type=int, default=16, help="DRAM 求和线程数（=CPU 专家线程）")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--nvme-file", default="/run/media/xpz/models/DeepSeek-V4.1-Flash-1017-official/model-00001-of-00048.safetensors")
    ap.add_argument("--chunk-mb", type=int, default=256, help="H2D 每次拷贝大小")
    ap.add_argument("--sm", action="store_true", help="同时让 GPU 跑 matmul")
    ap.add_argument("--stage", type=float, default=0.0,
                    help="模拟 ComfyUI 显存 staging 的目标 GB/s（文件→pinned→显存）")
    ap.add_argument("--stage-file", default="", help="staging 读取的文件（要求大于 1GB）")
    ap.add_argument("--stage-gpu", type=int, default=4, help="staging 拷到哪张卡（默认 4=ComfyUI 那张）")
    ap.add_argument("--stage-chunk-mb", type=int, default=64)
    ap.add_argument("--stage-churn-mb", type=int, default=1024,
                    help="每 4 轮申请/释放多少 MiB 显存，模拟 cudaMalloc 抖动；0=关闭")
    ap.add_argument("--stage-reserve-mb", type=int, default=0,
                    help="在 stage-gpu 上常驻多少 MiB 显存，模拟模型已驻留（如 32000）")
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--report-every", type=float, default=2.0)
    a = ap.parse_args()

    global STOP_AT
    STOP_AT = now() + a.duration
    gpus = [int(g) for g in a.gpus.split(",") if g.strip()]
    print(f"CONFIG dram={a.dram}GB/s nvme={a.nvme}GB/s h2d={a.h2d}GB/s stage={a.stage}GB/s "
          f"threads={a.threads} gpus={gpus} dur={a.duration}s sm={a.sm} stage_gpu={a.stage_gpu}", flush=True)

    meters = {"dram": Meter(), "nvme": Meter(), "h2d": Meter(), "stage": Meter()}
    target_ref = {"dram": a.dram * 2**30}
    threads = []

    if a.dram > 0 or a.dram_steps:
        buf = torch.empty(int(a.dram_gb * 2**30), dtype=torch.uint8)
        buf.fill_(1)  # 真分配、真落地，避免 lazy 页
        print(f"[dram] 工作集 {a.dram_gb:.1f} GiB 已落地", flush=True)
        t = threading.Thread(target=dram_worker,
                             args=(buf, meters["dram"], a.dram * 2**30, a.threads, target_ref),
                             daemon=True)
        t.start()
        threads.append(t)

    if a.dram_steps:
        def ramp() -> None:
            for s in a.dram_steps.split(","):
                if now() >= STOP_AT:
                    break
                gb = float(s.strip())
                target_ref["dram"] = gb * 2**30
                print(f"[step] dram={gb:g}GB/s", flush=True)
                try:
                    os.fsync(1)
                except OSError:
                    pass
                time.sleep(a.step_secs)

        t = threading.Thread(target=ramp, daemon=True)
        t.start()
        threads.append(t)

    if a.nvme > 0:
        t = threading.Thread(target=nvme_worker,
                             args=(a.nvme_file, meters["nvme"], a.nvme * 2**30, 1 << 20, 7), daemon=True)
        t.start()
        threads.append(t)

    if a.h2d > 0:
        t = threading.Thread(target=h2d_worker,
                             args=(gpus, meters["h2d"], a.h2d * 2**30, a.chunk_mb), daemon=True)
        t.start()
        threads.append(t)

    if a.sm:
        t = threading.Thread(target=sm_worker, args=(gpus, 4096), daemon=True)
        t.start()
        threads.append(t)

    if a.stage > 0:
        if not a.stage_file or not os.path.exists(a.stage_file):
            print(f"[stage] 文件不存在: {a.stage_file!r}", flush=True)
            return
        t = threading.Thread(
            target=stage_worker,
            args=(a.stage_file, meters["stage"], a.stage * 2**30, a.stage_chunk_mb * 2**20,
                  a.stage_gpu, a.stage_churn_mb * 2**20, a.stage_reserve_mb * 2**20), daemon=True)
        t.start()
        threads.append(t)

    if not threads:
        print("没有指定任何负载，退出", flush=True)
        return

    t_report = now()
    while now() < STOP_AT:
        time.sleep(0.5)
        if now() - t_report >= a.report_every:
            dt = now() - t_report
            t_report = now()
            line = f"[{time.strftime('%H:%M:%S')}] "
            for k, m in meters.items():
                with m.lock:
                    b, m.bytes = m.bytes, 0
                line += f"{k}={b / dt / 2**30:5.2f}GB/s "
            line += f"target_dram={target_ref['dram'] / 2**30:g}GB/s"
            print(line.strip(), flush=True)
            try:
                os.fsync(1)   # 死机也要把最后一行落到盘上
            except OSError:
                pass

    for t in threads:
        t.join(timeout=5)
    print("LOAD-DONE", flush=True)


if __name__ == "__main__":
    main()
