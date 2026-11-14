import glob
import json
import os
import subprocess
import threading
import time
import urllib.request

INTERVAL = 0.2
BASE = "http://127.0.0.1:9010"


def ranks():
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    found = {}
    for line in out.splitlines():
        if "Worker_PP" not in line:
            continue
        head = line.split(None, 1)
        try:
            pid = int(head[0])
        except ValueError:
            continue
        tail = head[1].split("Worker_PP")[-1].strip()
        if tail.isdigit():
            found[int(tail)] = pid
    return found


def proc_ticks(pid):
    with open("/proc/%d/stat" % pid) as handle:
        fields = handle.read().split()
    return int(fields[13]) + int(fields[14])


def thread_ticks(pid):
    values = {}
    for path in glob.glob("/proc/%d/task/*/stat" % pid):
        try:
            with open(path) as handle:
                fields = handle.read().split()
            values[int(fields[0])] = int(fields[13]) + int(fields[14])
        except OSError:
            continue
    return values


para = (
    "在分布式推理系统中，流水线并行把模型按层切分到多张卡上，每一段负责一部分层，"
    "段与段之间通过点对点通信传递激活值。"
)
nonce = os.environ.get("PROBE_NONCE", "cold-%d" % int(time.time() * 1000))
body = {
    "model": "deepseek-v4.1-flash-1017",
    "messages": [{"role": "user", "content": "编号 %s ：\n" % nonce + para * 2500}],
    "max_tokens": 8,
    "stream": False,
}

result = {}


def send():
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        started = time.time()
        raw = urllib.request.urlopen(req, timeout=600).read()
        result["ttft"] = time.time() - started
        usage = json.loads(raw).get("usage", {})
        result["prompt_tokens"] = usage.get("prompt_tokens")
    except Exception as exc:
        result["error"] = str(exc)


time.sleep(1.0)
pids = ranks()
base_rank = {rank: proc_ticks(pid) for rank, pid in pids.items()}
base_thread = {tid: val for tid, val in thread_ticks(pids[4]).items()}
started = time.time()
worker = threading.Thread(target=send)
worker.start()

peak = {rank: 0.0 for rank in pids}
busy_peak = 0
samples = 0
prev_rank = dict(base_rank)
prev_thread = dict(base_thread)
while worker.is_alive() and time.time() - started < 240:
    time.sleep(INTERVAL)
    samples += 1
    for rank, pid in pids.items():
        now = proc_ticks(pid)
        cores = (now - prev_rank[rank]) / 100.0 / INTERVAL
        peak[rank] = max(peak[rank], cores)
        prev_rank[rank] = now
    now_thread = thread_ticks(pids[4])
    busy = 0
    for tid, val in now_thread.items():
        if tid not in prev_thread:
            continue
        if (val - prev_thread[tid]) / 100.0 / INTERVAL > 0.5:
            busy += 1
    prev_thread = now_thread
    if busy > busy_peak:
        busy_peak = busy

worker.join()
print("samples=%d duration=%.1fs" % (samples, result.get("ttft", -1)))
print("prompt_tokens=%s" % result.get("prompt_tokens"))
print("peak cores per rank: " + "  ".join("PP%d=%.2f" % (r, peak[r]) for r in sorted(peak)))
print("PP4 busy threads peak (>0.5 core each): %d" % busy_peak)
if "error" in result:
    print("error:", result["error"])
