#!/usr/bin/env python3
"""持续压测引擎：W 路并发循环请求，直到 DUR 秒。

每条请求返回后打印 completion_tokens / tok/s，用于证明引擎"真的在算"，
而不是只有 HTTP 200。
"""
import json
import os
import sys
import threading
import time
import urllib.request

BASE = os.environ.get("ENGINE", "http://127.0.0.1:18000").rstrip("/")
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
W = int(sys.argv[2]) if len(sys.argv) > 2 else 6

with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
    MODEL = json.loads(r.read())["data"][0]["id"]
print(f"MODEL={MODEL} DUR={DUR} WORKERS={W}", flush=True)

stop_at = time.time() + DUR
lock = threading.Lock()
state = {"tok": 0, "req": 0, "err": 0}


def worker(w: int) -> None:
    n = 0
    while time.time() < stop_at:
        n += 1
        body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content":
                          f"Write a long detailed battle scene, chapter {w}-{n}, "
                          "with many characters and dialogue."}],
            "max_tokens": 400,
        }).encode()
        req = urllib.request.Request(
            BASE + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            c = d.get("usage", {}).get("completion_tokens", 0)
            dt = time.time() - t0
            with lock:
                state["tok"] += c
                state["req"] += 1
            print(f"w{w} r{n} {c}tok {c / dt if dt else 0:.1f}tok/s {dt:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            with lock:
                state["err"] += 1
            print(f"w{w} r{n} ERR {type(e).__name__} {e}", flush=True)


threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(1, W + 1)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"BUSY-DONE req={state['req']} err={state['err']} tok={state['tok']} "
      f"avg={state['tok'] / DUR if DUR else 0:.1f}tok/s", flush=True)
