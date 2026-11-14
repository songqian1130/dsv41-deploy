#!/usr/bin/env python3
"""DSV41-Flash PP5 固定基准测试 v1 —— 用例与口径均已锁定，不要修改本文件

为什么用服务端计数器而不是客户端流式时间戳
--------------------------------------------
实测发现（2026-11-14，ram 模式）：服务端 TTFT 0.378s / inter-token 23.1ms，
而客户端流式 chunk 的首块出现在 3.83s、29 个 chunk 装 116 token。
说明 detokenizer 会把多个 token 合并成一块并延后发送，**客户端 chunk 时间戳
既不能测 TTFT 也不能测 ITL**。因此本脚本以 /metrics 的引擎自报值为准：

    TTFT      = Δ vllm:time_to_first_token_seconds_sum / Δ ..._count
    decode    = Δ vllm:generation_tokens_total / Δ vllm:request_decode_time_seconds_sum
    ITL       = Δ vllm:inter_token_latency_seconds_sum / Δ ..._count
    聚合吞吐  = Δ generation_tokens_total / 墙钟时间（并发场景专用）

请求用非流式，避免 chunk 合并带来的偏差；墙钟时间同时记录，便于对照。

固定用例（改这里 = 换版本，必须同时升级 VERSION）
-------------------------------------------------
    C1_decode    200 词提示(约1.1k tok)，要长回答 400 tok   -> 测单流 decode tok/s
    C2_ttft22k   12000 词提示(约2.2万 tok)，只要一句话 64 tok -> 测冷启动 TTFT
    C3_conc3     3 路并发，各 2500 词(约4.6k tok)，300 tok    -> 测并发是否互相卡

prompt 每次用新 seed（BASE_SEED + run_index*10 + 用例序号）以避免前缀缓存命中，
长度/熵/提问/采样参数全部不变，所以不同 run_index 之间仍然完全可比。
cached_tokens>0 的样本会被标记 invalid（说明命中了缓存，不是冷启动）。

用法:  python3 bench_v1.py <mode标签> [run_index]
输出:  stdout + 本目录 results.jsonl（每行一次完整运行）
"""
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

VERSION = "bench_v1"
BASE = "http://127.0.0.1:9010"
ENDPOINT = BASE + "/v1/chat/completions"
METRICS = BASE + "/metrics"
MODEL = "deepseek-v4.1-flash-1017"
BASE_SEED = 100000
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")

Q_LONG = "\n\n请详细解释这段内容可能描述的技术原理，尽量展开，不少于三百字。"
Q_SHORT = "\n\n请用一句话概括上面的内容。"

CASES = [
    {"id": "C1_decode", "words": 200, "question": Q_LONG, "max_tokens": 400, "conc": 1,
     "note": "单流 decode 吞吐（短提示，长输出）"},
    {"id": "C2_ttft22k", "words": 12000, "question": Q_SHORT, "max_tokens": 64, "conc": 1,
     "note": "2.2万 token 冷启动 TTFT（engram/prefill 主指标）"},
    {"id": "C3_conc3", "words": 2500, "question": Q_LONG, "max_tokens": 300, "conc": 3,
     "note": "三路并发聚合吞吐与 TTFT 抖动"},
]

COUNTERS = [
    "generation_tokens_total",
    "prompt_tokens_total",
    "iteration_tokens_total_sum",
    "iteration_tokens_total_count",
    "inter_token_latency_seconds_sum",
    "inter_token_latency_seconds_count",
    "time_to_first_token_seconds_sum",
    "time_to_first_token_seconds_count",
    "request_decode_time_seconds_sum",
    "request_decode_time_seconds_count",
    "prefix_cache_hits_total",
    "prefix_cache_queries_total",
]


def make_prompt(seed, words, question):
    """高熵文本：汉字随机散列，n-gram 几乎不重复，保证 engram 真实读盘/读内存。"""
    rng = random.Random(seed)
    out = []
    for i in range(words):
        out.append(chr(0x4E00 + rng.randrange(0x5000)))
        if i % 12 == 11:
            out.append("。")
    return "".join(out) + question


def snap():
    raw = urllib.request.urlopen(METRICS, timeout=15).read().decode()
    got = {}
    for line in raw.splitlines():
        if not line.startswith("vllm:"):
            continue
        name = line.split("{")[0][5:]
        if name in COUNTERS:
            try:
                got[name] = got.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return got


def diff(a, b):
    return {k: b.get(k, 0.0) - a.get(k, 0.0) for k in set(a) | set(b)}


def one_call(prompt, max_tokens, timeout=3600):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": False}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    return r, time.time() - t0


def usage_of(r):
    return r.get("usage") or {}


def summarize(d, wall, n_req):
    """把计数器增量换算成指标。request_decode_time 缺失时退化为 ITL 估算。"""
    gen = d.get("generation_tokens_total", 0.0)
    dec = d.get("request_decode_time_seconds_sum", 0.0)
    itl_n = d.get("inter_token_latency_seconds_count", 0.0)
    itl_s = d.get("inter_token_latency_seconds_sum", 0.0)
    tft_n = d.get("time_to_first_token_seconds_count", 0.0)
    tft_s = d.get("time_to_first_token_seconds_sum", 0.0)
    out = {
        "gen_tokens": int(round(gen)),
        "wall_s": round(wall, 3),
        "tps_wall": round(gen / wall, 2) if wall > 0 else None,
        "ttft_s": round(tft_s / tft_n, 3) if tft_n else None,
        "decode_s": round(dec, 3) if dec else None,
        "tps_decode": round(gen / dec, 2) if dec > 0 else None,
        "itl_ms": round(itl_s / itl_n * 1000, 1) if itl_n else None,
    }
    return out


def run_case(case, seed, mode):
    label = f"{mode}/{case['id']}"
    # 并发用例每路用独立 seed：同一 prompt 并发会让第 2、3 路命中第 1 路刚建好的
    # 前缀缓存，TTFT 变成缓存命中值，测不到真实冷启动排队。
    prompts = [make_prompt(seed + i, case["words"], case["question"])
               for i in range(max(1, case["conc"]))]
    a = snap()
    t0 = time.time()
    if case["conc"] == 1:
        r, wall = one_call(prompts[0], case["max_tokens"])
        walls = [wall]
        usages = [usage_of(r)]
    else:
        with ThreadPoolExecutor(max_workers=case["conc"]) as ex:
            futs = [ex.submit(one_call, p, case["max_tokens"]) for p in prompts]
            res = [f.result() for f in futs]
        wall = time.time() - t0
        walls = [w for _, w in res]
        usages = [usage_of(r) for r, _ in res]
    wall = time.time() - t0
    d = diff(a, snap())
    m = summarize(d, wall, len(walls))
    cached = sum(((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                 for u in usages)
    row = {
        "case": case["id"], "label": label, "conc": case["conc"],
        "prompt_tokens": [u.get("prompt_tokens") for u in usages],
        "completion_tokens": [u.get("completion_tokens") for u in usages],
        "cached_tokens": cached,
        "per_request_wall_s": [round(w, 3) for w in walls],
        "valid_cold": cached == 0,
    }
    row.update(m)
    return row


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    run_index = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{VERSION}] mode={mode} run_index={run_index} at {stamp}", flush=True)
    rows = []
    for ci, case in enumerate(CASES):
        seed = BASE_SEED + run_index * 10 + ci
        row = run_case(case, seed, mode)
        rows.append(row)
        print("%-14s in=%s out=%s ttft=%s ts_decode=%s ts_wall=%s itl=%sms ready_wall=%ss cold=%s"
              % (case["id"], row["prompt_tokens"], row["completion_tokens"],
                 row["ttft_s"], row["tps_decode"], row["tps_wall"], row["itl_ms"],
                 row["per_request_wall_s"], row["valid_cold"]), flush=True)
        time.sleep(3)

    summary = {"version": VERSION, "mode": mode, "run_index": run_index,
               "ts": stamp, "cases": rows}
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
