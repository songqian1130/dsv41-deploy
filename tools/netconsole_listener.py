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


def logmsg(stream, text):
    """pythonw 下 sys.stdout/stderr 是 None，所有流写入必须走这里兜底"""
    try:
        stream.write(text + "\n")
        stream.flush()
    except Exception:
        pass


def logfile() -> str:
    return os.path.join(LOGDIR, "netconsole-%s.log" % time.strftime("%Y%m%d"))


def main() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # 绑定重试：若端口被占用（如其他实例先起了），等待而不是退出
    while True:
        try:
            s.bind(("0.0.0.0", PORT))
            break
        except OSError as e:
            logmsg(sys.stderr, "bind %d failed (%s), retry in 30s" % (PORT, e))
            time.sleep(30)
    logmsg(sys.stdout, "netconsole listener on UDP %d -> %s" % (PORT, LOGDIR))
    while True:
        # recv：Windows 上服务器侧 ICMP port-unreachable 会在下一次 recvfrom 抛
        # ConnectionResetError(10054)——吞掉继续收，绝不退出
        try:
            data, addr = s.recvfrom(65535)
        except OSError as e:
            logmsg(sys.stderr, "recv error %s" % e)
            time.sleep(0.2)
            continue
        # 写盘：文件被杀软/索引器短暂锁定等场景重试，不退出
        try:
            line = data.decode("utf-8", errors="replace").rstrip("\n")
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            with open(logfile(), "a", encoding="utf-8") as f:
                f.write("[%s] %s %s\n" % (stamp, addr[0], line))
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            logmsg(sys.stderr, "write error %s" % e)
            time.sleep(0.05)
            continue


if __name__ == "__main__":
    while True:
        try:
            main()
        except KeyboardInterrupt:
            break
        except Exception as e:
            # 顶层兜底：任何意外异常打日志后 3 秒重来，进程永活
            logmsg(sys.stderr, "fatal %r, restart in 3s" % e)
            time.sleep(3)
