#!/usr/bin/env python3
"""统一基准 runner（cases_v1）

设计要点
--------
1. **用例正文来自 cases_v1.json，永不随机生成**。脚本启动先校验 sha256，
   与文件内声明的值不符就直接退出——防止有人改了用例却复用旧结果。
2. **唯一变化的是 prompt 开头的 nonce**（形如 `[run0031-C1_decode]`），
   约 8~12 个 token，作用是让前缀缓存必然失效。正文一字不改，所以样本完全可比。
3. 指标全部取服务端 /metrics 计数器增量（客户端 chunk 时间戳不可靠，见 bench_v1 注释）。
4. 结果追加到 results.jsonl，每条都带 cases_sha256 + nonce，可反查。

用法
----
    python3 run_cases.py <label> [run_index] [--set bench|real|all]

    python3 run_cases.py ram 31                 # bench 三档（C1/C2/C3）
    python3 run_cases.py ram 31 --set real       # 真实负载四档
    python3 run_cases.py dspark 41 --set all     # 全部七档
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:9010"
ENDPOINT = BASE + "/v1/chat/completions"
METRICS = BASE + "/metrics"
MODEL = "deepseek-v4.1-flash-1017"
HERE = os.path.dirname(os.path.abspath(__file__))
CASES_FILE = os.path.join(HERE, "cases_v1.json")
RESULTS = os.path.join(HERE, "results.jsonl")

COUNTERS = [
    "generation_tokens_total",
    "prompt_tokens_total",
    "inter_token_latency_seconds_sum",
    "inter_token_latency_seconds_count",
    "time_to_first_token_seconds_sum",
    "time_to_first_token_seconds_count",
    "request_decode_time_seconds_sum",
    "request_decode_time_seconds_count",
    "prefix_cache_hits_total",
    "prefix_cache_queries_total",
]


def load_cases(strict=True):
    with open(CASES_FILE, encoding="utf-8") as fh:
        doc = json.load(fh)
    h = hashlib.sha256()
    for cid in sorted(doc["cases"]):
        for p in doc["cases"][cid]["prompts"]:
            h.update(p.encode("utf-8"))
        h.update(str(doc["cases"][cid]["max_tokens"]).encode())
    actual = h.hexdigest()
    if strict and actual != doc["sha256"]:
        sys.exit(
            "用例被改动过：声明 %s，实际 %s。\n"
            "如确需改用例，请新建 cases_v2.json 并升级版本号。"
            % (doc["sha256"], actual)
        )
    return doc, actual


def snap():
    raw = urllib.request.urlopen(METRICS, timeout=15).read().decode()
    got, spec = {}, {}
    for line in raw.splitlines():
        if not line.startswith("vllm:"):
            continue
        name = line.split("{")[0][5:]
        try:
            val = float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
        if name in COUNTERS:
            got[name] = got.get(name, 0.0) + val
        elif "spec_decode" in name:
            spec[name] = spec.get(name, 0.0) + val
    return got, spec


def diff(a, b):
    return {k: b.get(k, 0.0) - a.get(k, 0.0) for k in set(a) | set(b)}


def one_call(prompt, max_tokens, timeout=3600):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    return resp, time.time() - t0


def run_case(doc, cid, nonce_tag, label):
    case = doc["cases"][cid]
    conc = case["conc"]
    bodies = case["prompts"]
    assert len(bodies) >= conc, "%s 正文份数少于并发数" % cid

    prompts = [
        "[%s-%s-%d]\n%s" % (nonce_tag, cid, i, bodies[i]) for i in range(conc)
    ]

    m0, s0 = snap()
    t0 = time.time()
    if conc == 1:
        resp, wall = one_call(prompts[0], case["max_tokens"])
        walls, usages = [wall], [resp.get("usage") or {}]
    else:
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = [ex.submit(one_call, p, case["max_tokens"]) for p in prompts]
            got = [f.result() for f in futs]
        wall = time.time() - t0
        walls = [w for _, w in got]
        usages = [(r.get("usage") or {}) for r, _ in got]
    wall = time.time() - t0
    m1, s1 = snap()

    d = diff(m0, m1)
    ds = diff(s0, s1)
    gen = d.get("generation_tokens_total", 0.0)
    dec = d.get("request_decode_time_seconds_sum", 0.0)
    itl_n = d.get("inter_token_latency_seconds_count", 0.0)
    itl_s = d.get("inter_token_latency_seconds_sum", 0.0)
    tft_n = d.get("time_to_first_token_seconds_count", 0.0)
    tft_s = d.get("time_to_first_token_seconds_sum", 0.0)

    draft = ds.get("spec_decode_num_draft_tokens_total", 0.0)
    acc = ds.get("spec_decode_num_accepted_tokens_total", 0.0)
    cached = sum(
        ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        for u in usages
    )

    return {
        "case": cid,
        "kind": case["kind"],
        "label": "%s/%s" % (label, cid),
        "conc": conc,
        "prompt_tokens": [u.get("prompt_tokens") for u in usages],
        "completion_tokens": [u.get("completion_tokens") for u in usages],
        "cached_tokens": cached,
        "valid_cold": cached == 0,
        "wall_s": round(wall, 3),
        "per_request_wall_s": [round(w, 3) for w in walls],
        "ttft_s": round(tft_s / tft_n, 3) if tft_n else None,
        "tps_decode": round(gen / dec, 2) if dec > 0 else None,
        "tps_wall": round(gen / wall, 2) if wall > 0 else None,
        "itl_ms": round(itl_s / itl_n * 1000, 1) if itl_n else None,
        "draft_tokens": int(draft),
        "accepted_tokens": int(acc),
        "acc_per_draft": round(acc / draft, 3) if draft > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("run_index", nargs="?", type=int, default=1)
    ap.add_argument("--set", default="bench", choices=["bench", "real", "all"])
    args = ap.parse_args()

    doc, sha = load_cases()
    nonce_tag = "run%04d" % args.run_index
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    ids = [
        cid for cid in doc["cases"]
        if args.set == "all" or doc["cases"][cid]["kind"] == args.set
    ]
    print(
        "[cases_v1 %s] label=%s nonce=%s cases=%d sha=%s"
        % (doc["version"], args.label, nonce_tag, len(ids), sha[:12]),
        flush=True,
    )

    rows = []
    for cid in ids:
        row = run_case(doc, cid, nonce_tag, args.label)
        rows.append(row)
        print(
            "%-12s in=%s out=%s ttft=%s dec=%s wall=%s itl=%sms acc/draft=%s cold=%s"
            % (
                cid, row["prompt_tokens"], row["completion_tokens"], row["ttft_s"],
                row["tps_decode"], row["tps_wall"], row["itl_ms"],
                row["acc_per_draft"], row["valid_cold"],
            ),
            flush=True,
        )
        time.sleep(3)

    summary = {
        "version": doc["version"],
        "cases_sha256": sha,
        "nonce": nonce_tag,
        "label": args.label,
        "run_index": args.run_index,
        "ts": stamp,
        "cases": rows,
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
