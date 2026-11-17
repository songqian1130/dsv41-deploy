#!/usr/bin/env python3
"""解析 results.jsonl 中指定 label 的逐轮数据，输出均值对照。用法: parse_bench.py <label> [label2...]"""
import json, sys

PATH = "/home/xpz/dsv41-repo-test/bench/results.jsonl"
want = set(sys.argv[1:])
rows = [json.loads(l) for l in open(PATH) if l.strip()]
for label in want:
    sel = [r for r in rows if r.get("label") == label]
    print(f"=== {label}: {len(sel)} records ===")
    aw = ad = 0.0
    for r in sel:
        walls = [c["tps_wall"] for c in r["cases"]]
        decs = [c["tps_decode"] for c in r["cases"]]
        aw += sum(walls) / 4; ad += sum(decs) / 4
        print("run%s ts=%s avg_wall=%.2f avg_dec=%.2f walls=%s decs=%s" % (
            r["run_index"], r["ts"][11:19], sum(walls)/4, sum(decs)/4,
            [round(w,1) for w in walls], [round(d,1) for d in decs]))
    if sel:
        n = len(sel)
        print(f"==> {label} 均值: wall={aw/n:.2f} decode={ad/n:.2f} (n={n})")
