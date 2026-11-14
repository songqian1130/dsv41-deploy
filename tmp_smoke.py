#!/usr/bin/env python3
# B 冒烟测试 + 启动时间戳 + 速度采样
import json, subprocess, time, urllib.request

BASE = "http://127.0.0.1:9010/v1/chat/completions"

# 1) 冒烟
req = urllib.request.Request(BASE, data=json.dumps({
    "model": "deepseek-v4.1-flash-1017",
    "messages": [{"role": "user", "content": "用一句话回答：1+1等于几"}],
    "max_tokens": 60,
}).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=120) as r:
    d = json.loads(r.read())
m = d["choices"][0]["message"]
txt = (m.get("content") or m.get("reasoning_content") or "")[:100].replace("\n", " ")
u = d.get("usage", {})
print("冒烟:", txt)
print("prompt/completion:", u.get("prompt_tokens"), u.get("completion_tokens"),
      f"| 墙钟 {time.time()-t0:.1f}s (含 TTFT，冷启动首轮仅验证可用性)")

# 2) 单流速度（丢掉上面冷启动首轮后测 200 token）
req2 = urllib.request.Request(BASE, data=json.dumps({
    "model": "deepseek-v4.1-flash-1017",
    "messages": [{"role": "user", "content": "写一段 300 字左右的短文，介绍春天的特点。"}],
    "max_tokens": 200,
    "stream": True,
}).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
ntok, first = 0, None
with urllib.request.urlopen(req2, timeout=180) as r:
    for line in r:
        if not line.startswith(b"data:") or b"[DONE]" in line:
            continue
        try:
            j = json.loads(line[5:])
        except Exception:
            continue
        delta = j["choices"][0].get("delta", {})
        piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
        if piece:
            if first is None:
                first = time.time() - t0
            ntok += max(1, len(piece) // 3)  # 粗略 token 估计
total = time.time() - t0
if first and ntok:
    print(f"单流: 首 token {first:.2f}s, ~{ntok} tok / {total:.1f}s ≈ {ntok/(total-first):.1f} tok/s (粗估)")
else:
    print("单流采样失败")
