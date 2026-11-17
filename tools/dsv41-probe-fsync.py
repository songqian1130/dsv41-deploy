#!/usr/bin/env python3
"""bpftrace 输出逐行 fsync 落盘：死机时已写行保留（bpftrace 自身 stdout 是块缓冲，
死机即丢；经此管道每行 append + fsync 到磁盘）。"""
import os
import sys
import time

OUT = "/var/log/dsv41-blackbox/probe.log"


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fh = open(OUT, "a", buffering=1)
    for line in sys.stdin:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        body = line.strip()
        if body:
            fh.write("%s %s\n" % (ts, body))
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    # stdin 关闭（bpftrace 退出）
    fh.write("%s [probe-fsync] upstream closed\n" % time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        os.fsync(fh.fileno())
    except OSError:
        pass
    fh.close()


if __name__ == "__main__":
    main()
