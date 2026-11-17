#!/usr/bin/env python3
"""dsv41 NVML 金丝雀看门狗 —— 死锁触发的最后一张网

原理：NVML init（pynvml.nvmlInit）要打开 /dev/nvidia* 设备——一旦驱动锁死锁，
     它会卡在 D 状态。用带超时的子进程做金丝雀：
     - 正常：<1s 返回
     - 死锁：卡住 → 20s 超时 → 触发 sysrq 三连 dump（w/l/t），经 netconsole 外发

sysrq 含义：
  w = 打印所有 D 状态（不可中断阻塞）任务及其调用栈   → 卡死在哪个函数
  l = 打印所有 CPU 当前正在执行的函数                → 自旋/硬锁场景
  t = 打印全部任务的调度状态与调用栈                 → 完整调度现场

注意：金丝雀本身每 10s 一次设备 open/close（0.1Hz，远低于旧 blackbox 的 1Hz），
     这是有意保留的最小化"燃料"——因为金丝雀必须真的去 open 才能试出死锁。
"""
import subprocess
import time
import os
import sys

VENV_PY = "/home/xpz/venvs/comfy/bin/python"
CANARY = (
    "import pynvml, time; "
    "t = time.time(); "
    "pynvml.nvmlInit(); "
    "pynvml.nvmlShutdown(); "
    "print('canary-ok %.3fs' % (time.time() - t))"
)
LOG = "/var/log/dsv41-blackbox/tripwire.log"
TIMEOUT = 20          # 金丝雀超时 = 死锁判据
INTERVAL = 10         # 巡检间隔
MAX_STUCK_CHILDREN = 3  # 累积卡死子进程上限，超过后转被动模式

stuck_children = 0


def log(msg: str) -> None:
    line = "%s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
    try:
        with open(LOG, "a") as f:
            f.write(line)
            os.fsync(f.fileno())
    except OSError:
        pass
    sys.stdout.write(line)
    sys.stdout.flush()


def sysrq(ch: str, label: str) -> None:
    try:
        with open("/proc/sysrq-trigger", "w") as f:
            f.write(ch)
        log("sysrq %s (%s) triggered" % (ch, label))
    except OSError as e:
        log("sysrq %s failed: %s" % (ch, e))


def main() -> None:
    global stuck_children
    log("tripwire start (timeout=%ds interval=%ds)" % (TIMEOUT, INTERVAL))
    while True:
        try:
            r = subprocess.run(
                [VENV_PY, "-c", CANARY],
                capture_output=True, text=True, timeout=TIMEOUT,
            )
            if r.returncode == 0:
                out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
                if stuck_children:
                    log("canary recovered after %d stuck cycles" % stuck_children)
                    stuck_children = 0
                try:
                    secs = float(out.split()[-1].rstrip("s"))
                    if secs > 2.0:
                        log("SLOW canary: %s" % out)
                except (ValueError, IndexError):
                    pass
            else:
                log("canary rc=%d err=%s" % (r.returncode, r.stderr.strip()[-160:]))
        except subprocess.TimeoutExpired:
            stuck_children += 1
            log("*** CANARY STUCK x%d -> sysrq dump ***" % stuck_children)
            sysrq("w", "blocked-tasks(D-state stacks)")
            time.sleep(6)
            sysrq("l", "all-cpu-backtrace")
            time.sleep(6)
            sysrq("t", "all-tasks")
            if stuck_children >= MAX_STUCK_CHILDREN:
                log("passive mode: %d stuck children, stop spawning canary" % stuck_children)
                while True:
                    time.sleep(300)
                    log("passive heartbeat (stuck children=%d)" % stuck_children)
        except FileNotFoundError:
            log("venv python missing: %s" % VENV_PY)
            time.sleep(60)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
