#!/usr/bin/env python3
"""真实负载风格用例（real_case v1）

与 bench_v1 的区别：
- bench_v1 用高熵随机汉字，目的是绕开 n-gram 前缀缓存 -> 但它同时把 dspark 的
  投机接受率压到最低（实测 2.48），严重低估 dspark 的真实速度。
- real_case 用真实业务里最常见的四类输出（代码 / JSON / 中文技术解释 / 数学推导），
  可预测性远高于随机文本，用来测 dspark 在「真实分布」下的接受长度与 decode 速度。

用法：
    python3 real_case.py <label>

输出：每行一个用例的 JSON，同时追加到 results_real.jsonl
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:9010"
MODEL = "deepseek-v4.1-flash-1017"

CASES = [
    (
        "code_python",
        "用 Python 写一个线程安全的 LRU 缓存类，包含 get / put / delete 三个方法，"
        "带完整类型注解和 docstring，并附一段 pytest 单元测试。直接给完整代码，不要解释。",
        500,
    ),
    (
        "json_struct",
        "把下面的需求转成 JSON 输出：三个员工，字段为 name、role、level、skills（字符串数组），"
        "level 是 1 到 5 的整数。只输出 JSON 本体，不要任何说明文字。",
        400,
    ),
    (
        "zh_tech",
        "解释 vLLM 的 PagedAttention 原理，以及它和连续批处理（continuous batching）的关系，"
        "要具体到 KV cache 的内存布局层面，并举一个显存利用率提升的例子。",
        500,
    ),
    (
        "math_steps",
        "求 sum_{n=1}^{100} n^2 的闭式表达，并写出完整推导过程，每一步都要说明依据。",
        400,
    ),
]

KEYS = [
    "generation_tokens_total",
    "request_decode_time_seconds_sum",
    "time_to_first_token_seconds_sum",
    "time_to_first_token_seconds_count",
    "inter_token_latency_seconds_sum",
]


def scrape():
    """返回 (固定指标 dict, 所有 spec_decode 相关计数 dict)"""
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=15).read().decode()
    fixed, spec = {}, {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        if not line.startswith("vllm:"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0][5:]
        try:
            val = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        if name in KEYS:
            fixed[name] = fixed.get(name, 0.0) + val
        elif "spec_decode" in name:
            spec[name] = spec.get(name, 0.0) + val
    return fixed, spec


def post(prompt, max_tokens):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=900))
    elapsed = time.time() - t0
    usage = resp.get("usage", {})
    return elapsed, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "real"
    rows = []
    for name, prompt, max_tokens in CASES:
        f0, s0 = scrape()
        wall, ptok, ctok = post(prompt, max_tokens)
        f1, s1 = scrape()

        d = {k: f1.get(k, 0.0) - f0.get(k, 0.0) for k in KEYS}
        tps_decode = (
            d["generation_tokens_total"] / d["request_decode_time_seconds_sum"]
            if d["request_decode_time_seconds_sum"] > 0
            else float("nan")
        )
        itl_ms = (
            d["inter_token_latency_seconds_sum"]
            / max(d["generation_tokens_total"], 1)
            * 1000
        )
        draft = s1.get("spec_decode_num_draft_tokens_total", 0.0) - s0.get(
            "spec_decode_num_draft_tokens_total", 0.0
        )
        acc = s1.get("spec_decode_num_accepted_tokens_total", 0.0) - s0.get(
            "spec_decode_num_accepted_tokens_total", 0.0
        )
        acc_len = (acc / draft + 1.0) if draft > 0 else None

        row = {
            "label": f"{label}/{name}",
            "case": name,
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
            "wall_s": round(wall, 3),
            "tps_wall": round(ctok / wall, 2) if wall > 0 else None,
            "tps_decode": round(tps_decode, 2) if tps_decode == tps_decode else None,
            "itl_ms": round(itl_ms, 1),
            "accept_len": round(acc_len, 2) if acc_len else None,
            "draft_tokens": int(draft),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    with open("results_real.jsonl", "a") as fh:
        fh.write(json.dumps({"label": label, "ts": time.strftime("%FT%T"), "rows": rows},
                            ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
