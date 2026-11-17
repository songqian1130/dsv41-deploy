#!/usr/bin/env python3
"""dsv41 黑匣子 v2：pynvml 常驻连接版（无 open/close 循环）。

背景：v1 每秒 subprocess.run(nvidia-smi)——每次 = NVML init/deinit = 一对设备
open/close，是 NVIDIA open 驱动 #968 族死锁（nv_open_q × close 互等 rwsem）的竞态
燃料。v2 用 pynvml 一次 nvmlInit() 长连接，循环内只做 NVML 查询 ioctl，不产生
设备 open/close 事件。CSV 列格式与 v1 完全兼容（nvsmi_ms 列改名 nvml_ms，
按天分文件新 header）。

每行写完立即 fsync —— 硬冻结时 page cache 里的数据会丢，fsync 过的不会。
"""
import os
import threading
import time
from datetime import datetime

import pynvml

OUTDIR = "/var/log/dsv41-blackbox"
KEEP_DAYS = 4
NGPU = 5
os.makedirs(OUTDIR, exist_ok=True)


def now_iso():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def day_path():
    return os.path.join(OUTDIR, "blackbox-%s.csv" % time.strftime("%Y%m%d"))


FIELDS = ("pwr", "temp", "util", "mem", "smclk", "gen", "thr")
HEADER = ",".join(
    ["ts", "iso"]
    + ["g%d_%s" % (i, k) for i in range(NGPU) for k in FIELDS]
    + ["pkg0_w", "pkg1_w", "cpu_idle", "load1", "memavail_gb",
       "psi_cpu10", "psi_io10", "psi_mem10", "nvml_ms", "fsync_ms"]
)


def read_file(p, default=""):
    try:
        with open(p) as f:
            return f.read().strip()
    except Exception:
        return default


def rapl_uj(dom):
    v = read_file("/sys/class/powercap/intel-rapl:%d/energy_uj" % dom)
    try:
        return int(v)
    except Exception:
        return None


def psi(res):
    try:
        with open("/proc/pressure/" + res) as f:
            for tok in f.readline().split():
                if tok.startswith("avg10="):
                    return float(tok[6:])
    except Exception:
        pass
    return 0.0


def cpu_tick():
    try:
        with open("/proc/stat") as f:
            nums = [int(x) for x in f.readline().split()[1:]]
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        return idle, sum(nums)
    except Exception:
        return 0, 0


def memavail_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 1048576.0, 1)
    except Exception:
        pass
    return 0.0


def rotate():
    cutoff = time.time() - KEEP_DAYS * 86400
    for f in os.listdir(OUTDIR):
        p = os.path.join(OUTDIR, f)
        try:
            if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                os.unlink(p)
        except Exception:
            pass


def kmsg_loop():
    path = os.path.join(OUTDIR, "kmsg.log")
    while True:
        out = None
        fd = None
        try:
            out = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
            os.write(out, ("%s\t--BOOT--\n" % now_iso()).encode())
            os.fsync(out)
            fd = os.open("/dev/kmsg", os.O_RDONLY | os.O_NONBLOCK)
            try:
                os.lseek(fd, 0, os.SEEK_END)
            except OSError:
                pass
            while True:
                try:
                    data = os.read(fd, 65536)
                except BlockingIOError:
                    time.sleep(0.15)
                    continue
                if not data:
                    time.sleep(0.15)
                    continue
                rec = data.decode("utf-8", "replace").strip()
                os.write(out, ("%s\t%s\n" % (now_iso(), rec)).encode())
                os.fsync(out)
        except Exception:
            time.sleep(2)
        finally:
            for x in (fd, out):
                try:
                    if x is not None:
                        os.close(x)
                except Exception:
                    pass


def gpu_sample(h):
    """一次 NVML 查询（长连接内 ioctl，无设备 open/close）。返回 5 卡 × 7 字段。"""
    rows = []
    for i in range(NGPU):
        try:
            d = h[i]
            pwr = round(pynvml.nvmlDeviceGetPowerUsage(d) / 1000.0, 1)
            temp = pynvml.nvmlDeviceGetTemperature(d, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(d).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(d).used // 1048576
            smclk = pynvml.nvmlDeviceGetClockInfo(d, pynvml.NVML_CLOCK_SM)
            gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(d)
            thr = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(d) & 0xFFFFFFFF
            rows.append([pwr, temp, util, mem, smclk, gen, thr])
        except pynvml.NVMLError:
            rows.append(["E"] * 7)
    return rows


def main():
    threading.Thread(target=kmsg_loop, daemon=True).start()

    pynvml.nvmlInit()  # 一次性 init，进程生命周期内不再 open/close
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(NGPU)]

    prev_idle, prev_total = cpu_tick()
    p0, p1 = rapl_uj(0), rapl_uj(1)
    prev_t = time.monotonic()
    fh = None
    cur_day = None

    while True:
        loop_start = time.monotonic()
        try:
            today = time.strftime("%Y%m%d")
            if fh is None or today != cur_day:
                if fh:
                    try:
                        fh.close()
                    except Exception:
                        pass
                newfile = day_path()
                fresh = (not os.path.exists(newfile)) or os.path.getsize(newfile) == 0
                fh = open(newfile, "a", buffering=1)
                if fresh:
                    fh.write(HEADER + "\n")
                cur_day = today
                rotate()

            t_nv = time.perf_counter()
            gpus = gpu_sample(handles)
            nvml_ms = round((time.perf_counter() - t_nv) * 1000, 1)

            idle, total = cpu_tick()
            di, dt_ = idle - prev_idle, total - prev_total
            cpu_idle = round(100.0 * di / dt_, 1) if dt_ > 0 else 0.0
            prev_idle, prev_total = idle, total

            c0, c1 = rapl_uj(0), rapl_uj(1)
            now = time.monotonic()
            span = max(now - prev_t, 0.001)
            w0 = round((c0 - p0) / 1e6 / span, 1) if (c0 and p0 and c0 >= p0) else 0.0
            w1 = round((c1 - p1) / 1e6 / span, 1) if (c1 and p1 and c1 >= p1) else 0.0
            p0, p1, prev_t = c0, c1, now

            load1 = (read_file("/proc/loadavg", "0 0 0").split() or ["0"])[0]
            row = [str(int(time.time())), now_iso()]
            for g in gpus:
                row.extend(str(x) for x in g)
            row += [w0, w1, cpu_idle, load1, memavail_gb(),
                    psi("cpu"), psi("io"), psi("memory"), nvml_ms]

            t0 = time.perf_counter()
            fh.write(",".join(str(x) for x in row) + ",")
            fh.flush()
            os.fsync(fh.fileno())
            ms = round((time.perf_counter() - t0) * 1000, 2)
            fh.write("%.2f\n" % ms)
            fh.flush()
            os.fsync(fh.fileno())
        except Exception:
            pass

        elapsed = time.monotonic() - loop_start
        time.sleep(max(0.0, 1.0 - elapsed))


if __name__ == "__main__":
    main()
