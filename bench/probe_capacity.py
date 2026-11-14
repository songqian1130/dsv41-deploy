#!/usr/bin/env python3
"""容量 / 并发 / prefill 探针（独立于 cases_v1，不动冻结的速度用例集）

三种测量，结果追加到 capacity.jsonl：

  conc     N 路并发（默认 4）：单流 tok/s、聚合 tok/s、ITL、preemption 增量、KV 峰值
  prefill  prefill 斜率：多点 (prompt_tokens, ttft) 最小二乘 → tok/s（扣除固定开销）
  table    只算容量表：KV 池聚合 token ÷ 每会话上下文 = 最大不 preempt 会话数

用法：
  python3 probe_capacity.py conc    --label pp5fast --conc 4 --ctx 2000 --out 300
  python3 probe_capacity.py prefill --label pp5fast --sizes 20,80,240
  python3 probe_capacity.py table   --kv-tokens 131072 --seqs 4
"""
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:9010")
MODEL = os.environ.get("BENCH_MODEL", "deepseek-v4.1-flash-1017")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "capacity.jsonl")

WORDS = (
    "系统 推理 流水线 并行 张量 激活 通信 延迟 带宽 缓存 专家 路由 权重 量化 编译 调度 "
    "内存 显存 内核 线程 队列 批次 采样 概率 分布 训练 微调 蒸馏 剪枝 稀疏 稠密 矩阵 向量 "
    "注意 归约 归一 残差 梯度 优化 收敛 泛化 服务 部署 容器 镜像 端口 协议 网络 交换 光纤 "
    "存储 磁盘 吞吐 抖动 上下文 窗口 词表 编码 解码 门控 卷积 池化 归一化 正则 稀疏化 "
    "集群 节点 拓扑 链路 帧 报文 校验 重传 拥塞 分片 索引 分页 页表 调度器 抢占 公平 配额"
).split()

COUNTERS = [
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:generation_tokens_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
]


def rnd_text(n_words, seed):
    rng = random.Random(seed)
    return "".join(rng.choice(WORDS) for _ in range(n_words)) + "。"


def read_metrics():
    out = {}
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
            for line in r.read().decode("utf-8", "ignore").splitlines():
                if line.startswith("#") or " " not in line:
                    continue
                key, val = line.rsplit(" ", 1)
                name = key.split("{")[0]
                if name in COUNTERS:
                    try:
                        out[name] = out.get(name, 0.0) + float(val)
                    except ValueError:
                        pass
    except Exception:
        pass
    return out


def kv_gauge():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=5) as r:
            for line in r.read().decode("utf-8", "ignore").splitlines():
                if line.startswith("vllm:kv_cache_usage_perc"):
                    return float(line.rsplit(" ", 1)[1])
    except Exception:
        return None
    return None


class KvSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.halt = False
        self.peak = 0.0
        self.n = 0

    def run(self):
        while not self.halt:
            v = kv_gauge()
            if v is not None:
                self.peak = max(self.peak, v)
                self.n += 1
            time.sleep(0.2)


def stream_once(prompt, max_tokens):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    t_first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    if t_first is None:
                        t_first = time.time()
    t_end = time.time()
    return {
        "ttft": (t_first - t0) if t_first else float("nan"),
        "total": t_end - t0,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def delta(before, after, name):
    return after.get(name, 0.0) - before.get(name, 0.0)


def emit(rec):
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("SUMMARY " + json.dumps(rec, ensure_ascii=False))


def cmd_conc(args):
    label = args.get("label", "unnamed")
    conc = int(args.get("conc", 4))
    ctx_words = int(args.get("ctx", 2000))
    out_tok = int(args.get("out", 300))
    seed = args.get("seed") or str(random.randint(10000, 99999))

    before = read_metrics()
    sampler = KvSampler()
    sampler.start()

    prompts = [
        "[job %s-%d] " % (seed, i) + rnd_text(ctx_words, "%s-%d" % (seed, i))
        for i in range(conc)
    ]

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(stream_once, p, out_tok) for p in prompts]
        res = [f.result() for f in futs]
    wall = time.time() - t_start

    sampler.halt = True
    sampler.join(timeout=2)
    after = read_metrics()

    per_stream = []
    itls = []
    for r in res:
        dec = r["completion_tokens"] / (r["total"] - r["ttft"]) if r["total"] > r["ttft"] > 0 else 0.0
        itl = ((r["total"] - r["ttft"]) * 1000.0 / max(1, r["completion_tokens"] - 1))
        per_stream.append(round(dec, 1))
        itls.append(round(itl, 1))

    total_tokens = sum(r["completion_tokens"] for r in res)
    agg = total_tokens / wall if wall > 0 else 0.0
    pq = delta(before, after, "vllm:prefix_cache_queries_total")
    ph = delta(before, after, "vllm:prefix_cache_hits_total")
    ndraft = delta(before, after, "vllm:spec_decode_num_drafts_total")
    ndtok = delta(before, after, "vllm:spec_decode_num_draft_tokens_total")
    nacc = delta(before, after, "vllm:spec_decode_num_accepted_tokens_total")
    steps = ndraft
    tps_per_step = (nacc + steps) / steps if steps else None

    rec = {
        "kind": "conc",
        "label": label,
        "conc": conc,
        "ctx_words": ctx_words,
        "out_tokens": out_tok,
        "prompt_tokens": [r["prompt_tokens"] for r in res],
        "ttft_s": [round(r["ttft"], 3) for r in res],
        "per_stream_tps": per_stream,
        "per_stream_tps_mean": round(statistics.mean(per_stream), 1),
        "aggregate_tps": round(agg, 1),
        "itl_ms": itls,
        "itl_ms_mean": round(statistics.mean(itls), 1),
        "wall_s": round(wall, 2),
        "preemptions": int(delta(before, after, "vllm:num_preemptions_total")),
        "kv_peak": round(sampler.peak, 3),
        "kv_samples": sampler.n,
        "prefix_cache_hit_pct": round(100.0 * ph / pq, 2) if pq else None,
        "spec_drafts": int(ndraft),
        "spec_draft_tokens": int(ndtok),
        "spec_accepted": int(nacc),
        "spec_acc_per_draft": round(nacc / ndraft, 3) if ndraft else None,
        "tokens_per_step": round(tps_per_step, 2) if tps_per_step else None,
    }
    emit(rec)

    print("\n%d 路并发  ctx=%d words  out=%d tok  wall=%.1fs" % (conc, ctx_words, out_tok, wall))
    print("  单流 tok/s   %s  均值 %.1f" % (per_stream, rec["per_stream_tps_mean"]))
    print("  聚合 tok/s   %.1f" % rec["aggregate_tps"])
    print("  ITL ms       %s  均值 %.1f" % (itls, rec["itl_ms_mean"]))
    print("  KV 峰值      %.1f%%   采样 %d 次" % (rec["kv_peak"] * 100, sampler.n))
    print("  preemption   %d 次" % rec["preemptions"])
    print("  prefix 命中  %s" % (("%.2f%%" % rec["prefix_cache_hit_pct"]) if rec["prefix_cache_hit_pct"] is not None else "n/a"))
    print("  投机解码     drafts=%d 接受=%.2f/draft 每步=%.2f token" % (rec["spec_drafts"], rec["spec_acc_per_draft"] or 0, rec["tokens_per_step"] or 0))


def cmd_prefill(args):
    label = args.get("label", "unnamed")
    sizes = [int(x) for x in (args.get("sizes") or "20,80,240").split(",")]
    seed = str(random.randint(10000, 99999))

    before = read_metrics()
    points = []
    print("%8s %10s %9s %14s" % ("paras", "prompt_tok", "TTFT(s)", "TTFT/tok(ms)"))
    for n in sizes:
        prompt = "[trace %s-%d] " % (seed, n) + rnd_text(n * 271 // 1, "%s-%d" % (seed, n))
        r = stream_once(prompt, max_tokens=1)
        pt = r["prompt_tokens"] or (n * 271)
        pts = pt / r["ttft"] if r["ttft"] else float("nan")
        points.append({"paras": n, "prompt_tokens": pt, "ttft_s": round(r["ttft"], 3), "naive_tps": round(pts, 0)})
        print("%8d %10d %9.2f %14.2f" % (n, pt, r["ttft"], 1000.0 * r["ttft"] / pt), flush=True)
        time.sleep(0.5)
    after = read_metrics()

    n = len(points)
    mx = statistics.mean(p["prompt_tokens"] for p in points)
    my = statistics.mean(p["ttft_s"] for p in points)
    num = sum((p["prompt_tokens"] - mx) * (p["ttft_s"] - my) for p in points)
    den = sum((p["prompt_tokens"] - mx) ** 2 for p in points)
    slope = num / den if den else float("nan")
    intercept = my - slope * mx
    rate = 1.0 / slope if slope and slope > 0 else float("nan")

    rec = {
        "kind": "prefill",
        "label": label,
        "points": points,
        "slope_s_per_tok": round(slope, 8),
        "prefill_tps_slope": round(rate, 0),
        "fixed_overhead_s": round(intercept, 3),
        "prompt_tokens_total": int(delta(before, after, "vllm:prompt_tokens_total")),
        "cached_tokens": int(delta(before, after, "vllm:prompt_tokens_cached_total")),
    }
    emit(rec)
    print("\nslope = %.2e s/token  →  prefill %.0f tok/s（冷·随机文本，已扣固定开销）" % (slope, rate))
    print("固定开销（截距）%.2f s" % intercept)


def cmd_table(args):
    kv = int(args.get("kv-tokens", 131072))
    seqs = int(args.get("seqs", 4))
    print("KV 池 %d 聚合 token，max_num_seqs=%d\n" % (kv, seqs))
    print("%12s %14s %14s" % ("每会话上下文", "KV 理论上限", "实际可用会话"))
    for c in (4096, 8192, 16384, 32768, 65536, 131072):
        print("%12s %14d %14d" % ("%d k" % (c // 1024), kv // c, min(seqs, kv // c)))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    args = {}
    rest = sys.argv[2:]
    i = 0
    while i < len(rest):
        if rest[i].startswith("--"):
            k = rest[i][2:]
            if i + 1 < len(rest) and not rest[i + 1].startswith("--"):
                args[k] = rest[i + 1]
                i += 2
            else:
                args[k] = True
                i += 1
        else:
            i += 1
    {"conc": cmd_conc, "prefill": cmd_prefill, "table": cmd_table}[cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
