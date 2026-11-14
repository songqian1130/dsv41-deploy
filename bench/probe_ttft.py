#!/usr/bin/env python3
"""TTFT probe: cold prefill at several prompt lengths.

Adds a random nonce so prefix caching never hits. Reports:
  - TTFT (time to first content token)
  - prefill throughput = prompt_tokens / TTFT  (upper bound, includes launch overhead)
  - decode throughput  = completion_tokens / (end - ttft)
"""
import json
import os
import random
import sys
import time
import urllib.request

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:9010")
MODEL = os.environ.get("BENCH_MODEL", "deepseek-v4.1-flash-1017")

PARA = (
    "在分布式推理系统中，流水线并行把模型按层切分到多张卡上，每一段负责一部分层，"
    "段与段之间通过点对点通信传递激活值。当单个请求的顺序执行无法被掩盖时，"
    "流水线的气泡就会直接体现在首 token 延迟上。预填充阶段要把整个提示词一次算完，"
    "因此它的耗时随提示词长度近似线性增长，而注意力部分还带有二次项。"
)

WORDS = (
    "系统 推理 流水线 并行 张量 激活 通信 延迟 带宽 缓存 专家 路由 权重 量化 编译 调度 "
    "内存 显存 内核 线程 队列 批次 采样 概率 分布 训练 微调 蒸馏 剪枝 稀疏 稠密 矩阵 向量 "
    "注意 归约 归一 残差 梯度 优化 收敛 泛化 服务 部署 容器 镜像 端口 协议 网络 交换 光纤 "
    "存储 磁盘 吞吐 抖动 队列 上下文 窗口 词表 编码 解码 门控 卷积 池化 归一化 正则 稀疏化 "
    "集群 节点 拓扑 链路 帧 报文 校验 重传 拥塞 分片 索引 分页 页表 调度器 抢占 公平 配额"
).split()


def build_prompt(paras: int, nonce: str) -> str:
    return f"[trace {nonce}] " + (PARA * paras)


def build_prompt_diverse(paras: int, nonce: str) -> str:
    rng = random.Random(nonce)
    lines = [f"[trace {nonce}]"]
    for _ in range(paras):
        n = rng.randint(24, 38)
        lines.append("".join(rng.choice(WORDS) for _ in range(n)) + "。")
    return "\n".join(lines)


def stream_once(prompt: str, max_tokens: int) -> dict:
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
    out_chars = 0
    with urllib.request.urlopen(req, timeout=1800) as resp:
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
                piece = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or ""
                )
                if piece:
                    if t_first is None:
                        t_first = time.time()
                    out_chars += len(piece)
    t_end = time.time()
    return {
        "ttft": (t_first - t0) if t_first else float("nan"),
        "total": t_end - t0,
        "usage": usage,
        "chars": out_chars,
    }


def main() -> None:
    mode = "rep"
    args = sys.argv[1:]
    if args and args[0] in ("rep", "div"):
        mode = args.pop(0)
    sizes = [int(x) for x in (args or ["20", "80", "240", "360"])]
    seed = random.randint(10000, 99999)
    build = build_prompt_diverse if mode == "div" else build_prompt
    print(f"mode={mode}")
    print(f"{'paras':>7} {'prompt_tok':>10} {'TTFT(s)':>9} {'prefill tok/s':>14} {'decode tok/s':>13}")
    for n in sizes:
        nonce = f"{seed}-{n}-{random.randint(1000, 9999)}"
        prompt = build(n, nonce)
        r = stream_once(prompt, max_tokens=32)
        pt = r["usage"].get("prompt_tokens", 0)
        ct = r["usage"].get("completion_tokens", 0)
        pre = pt / r["ttft"] if r["ttft"] and pt else float("nan")
        dec = ct / (r["total"] - r["ttft"]) if ct and r["total"] > r["ttft"] else float("nan")
        print(f"{n:>7}k {pt:>10} {r['ttft']:>9.2f} {pre:>14.0f} {dec:>13.1f}", flush=True)


if __name__ == "__main__":
    main()
