#!/usr/bin/env python3
"""dsv41 平台脉搏采样器 —— 两个使命

1. SMI 计数（MSR 0x34，全 CPU 求和）：SMI（系统管理中断）由 BIOS/ME 发出，内核看不见、
   NMI 都可能被它吃掉。19:12 死机前 "perf interrupt took too long" 持续恶化
   （PMI 处理耗时 2.5k→10k 周期）是典型 SMI 干扰签名。死机后看 smi 曲线是否陡增。
2. 功耗墙看门：每次开机后 GPU1/2 会神秘变 110W、GPU4 变 250W（/dev/nvidia* 是 666，
   任何本地进程可用 pynvml 改、不留日志）。一旦检测到 PL 变化，立即抓 ps 快照当场。

pynvml 常驻连接（init 一次），无 nvidia-smi 子进程循环；MSR 读 /dev/cpu/*/msr 极轻。
"""
import os
import time
import subprocess

import pynvml

OUT = "/var/log/dsv41-blackbox/pulse.log"
INTERVAL = 5


def log(msg: str) -> None:
    line = "%s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
    try:
        with open(OUT, "a") as f:
            f.write(line)
            os.fsync(f.fileno())
    except OSError:
        pass
    sys_write(line)


def sys_write(s: str) -> None:
    try:
        sys.stdout.write(s)
        sys.stdout.flush()
    except Exception:
        pass


def read_smi_count() -> int:
    total = 0
    try:
        cpus = [d for d in os.listdir("/dev/cpu") if d.isdigit()]
        for c in cpus:
            try:
                with open("/dev/cpu/%s/msr" % c, "rb") as f:
                    f.seek(0x34)
                    raw = f.read(8)
                    total += int.from_bytes(raw, "little")
            except OSError:
                continue
    except OSError:
        return -1
    return total


def main() -> None:
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(5)]
    prev_pl = None
    log("pulse start (interval=%ds)" % INTERVAL)
    prev_smi = read_smi_count()
    while True:
        try:
            pls = []
            for i, h in enumerate(handles):
                try:
                    pls.append(round(pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0, 1))
                except pynvml.NVMLError:
                    pls.append(-1)
            smi = read_smi_count()
            delta = (smi - prev_smi) if (smi >= 0 and prev_smi >= 0) else -1
            prev_smi = smi
            log("smi_total=%s smi_5s=%s pl=%s" % (smi, delta, pls))
            cur = tuple(pls)
            if prev_pl is not None and cur != prev_pl:
                log("*** PL-CHANGED %s -> %s ***" % (list(prev_pl), list(cur)))
                try:
                    ps = subprocess.run(
                        ["ps", "-eo", "pid,user,etimes,args"],
                        capture_output=True, text=True, timeout=10,
                    ).stdout
                    keep = [
                        ln for ln in ps.splitlines()
                        if ("nvml" in ln.lower() or "nvidia" in ln.lower()
                            or "python" in ln.lower() or "PID" in ln)
                    ]
                    log("PL-CHANGED procs: %s" % " | ".join(keep[:15]))
                except Exception as e:
                    log("ps snapshot failed: %s" % e)
            prev_pl = cur
        except pynvml.NVMLError as e:
            log("nvml error: %s" % e)
            try:
                pynvml.nvmlInit()
                handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(5)]
            except Exception:
                pass
        except Exception as e:
            log("pulse error: %r" % e)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
