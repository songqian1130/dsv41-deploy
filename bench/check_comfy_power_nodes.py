#!/usr/bin/env python3
"""检查 ComfyUI 历史 workflow 里是否含功耗设定节点（查功耗墙乱源）"""
import json
import urllib.request

req = urllib.request.Request(
    "http://127.0.0.1:8188/history?max_items=10", headers={"Accept": "application/json"}
)
h = json.load(urllib.request.urlopen(req, timeout=5))
print("history entries:", len(h))
for k, v in h.items():
    p = v.get("prompt", [None, None, {}])
    graph = p[2] if len(p) > 2 else {}
    nodes = []
    if isinstance(graph, dict):
        for w in graph.values():
            if isinstance(w, dict):
                nodes.append(w.get("class_type", "?"))
    hits = [n for n in nodes if "ower" in n or "limit" in n.lower() or "PL" in n]
    st = v.get("status", {}).get("status_str", "?")
    print(k[:10], st, "power-nodes:", hits if hits else "none", "| all:", nodes[:6])

print("===crystools-power-grep===")
