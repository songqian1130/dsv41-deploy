#!/usr/bin/env python3
"""netconsole UDP 接收器（Windows 侧常驻）

接收服务器内核 printk 的 UDP 转发（端口 6667），逐条带接收时间戳落盘。
死机时 journald/磁盘都挂了，内核日志从这里活着流出——这就是"猝死型死机的黑匣子"。

用法: python netconsole_listener.py [端口，默认 6667]
日志: dsv41-deploy/logs/netconsole-YYYYMMDD.log
开机自启: schtasks ONLOGON（见部署记录）
"""
import socket
import sys
import os
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 6667
LOGDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs"))
os.makedirs(LOGDIR, exist_ok=True)


def logfile() -> str:
    return os.path.join(LOGDIR, "netconsole-%s.log" % time.strftime("%Y%m%d"))


def main() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", PORT))
    sys.stdout.write("netconsole listener on UDP %d -> %s\n" % (PORT, LOGDIR))
    sys.stdout.flush()
    while True:
        data, addr = s.recvfrom(65535)
        line = data.decode("utf-8", errors="replace").rstrip("\n")
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(logfile(), "a", encoding="utf-8") as f:
            f.write("[%s] %s %s\n" % (stamp, addr[0], line))
            f.flush()
            os.fsync(f.fileno())


if __name__ == "__main__":
    main()
