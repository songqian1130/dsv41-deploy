#!/usr/bin/env python3
"""取 ComfyUI 历史任务的工作流构成，并把最新完成的任务的完整 graph 导出为可重提交的 JSON。
用法: fetch_comfy_history.py [导出文件路径]  默认导出到 /home/xpz/manju/replay_workflow.json
"""
import json
import sys
import urllib.request

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/xpz/manju/replay_workflow.json"
h = json.load(urllib.request.urlopen("http://127.0.0.1:8188/history?max_items=8", timeout=10))
items = list(h.items())
print(f"history items: {len(items)}")
graph = None
for k, v in items[:8]:
    pr = v.get("prompt", [])
    g = pr[2] if len(pr) > 2 else {}
    types = [w.get("class_type", "?") for w in g.values()] if isinstance(g, dict) else []
    st = v.get("status", {}).get("status_str", "?")
    print(k[:14], st, "|", ",".join(types[:12]))
    if st == "success" and graph is None and g:
        graph = g
if graph:
    json.dump({"prompt": graph}, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("exported latest success graph ->", OUT)
else:
    print("no success graph found")
