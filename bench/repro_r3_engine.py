#!/usr/bin/env python3
"""R3 引擎侧：周期性打 24k token 长 prompt（大 prefill H2D + 集中专家补货 DMA），贴 Claude 生产形态。
用法: repro_r3_engine.py [持续秒] [间隔秒]   默认 300 15
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:9010"
MODEL = "deepseek-v4.1-flash-1017"
DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

# 构造 ~24k token 的长 prompt：技术文本重复 + 编号
base_text = open("/home/xpz/manju/synthetic_load.py", encoding="utf-8").read()
prompt = ""
i = 0
while len(prompt) < 70000:  # ~24k token
    prompt += f"\n[SECTION {i}] 分析下面这段代码的第 {i} 个潜在问题并给出修复建议：\n{base_text}\n"
    i += 1

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": prompt + "\n\n请只用一句话总结你的整体结论。"}],
    "max_tokens": 100,
    "temperature": 0,
    "stream": False,
}

n = 0
t0 = time.time()
while time.time() - t0 < DURATION:
    n += 1
    t_req = time.time()
    try:
        req = urllib.request.Request(
            BASE + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            body = json.loads(r.read())
        usage = body.get("usage", {})
        print(f"[r3] req{n} pt={usage.get('prompt_tokens')} ct={usage.get('completion_tokens')} "
              f"wall={time.time()-t_req:.1f}s @{time.strftime('%H:%M:%S')}", flush=True)
    except Exception as e:
        print(f"[r3] req{n} FAILED: {e}", flush=True)
    try:
        import os
        os.fsync(1)
    except OSError:
        pass
    time.sleep(INTERVAL)
print(f"[r3] DONE reqs={n}", flush=True)
