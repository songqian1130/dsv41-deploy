#!/usr/bin/env python3
"""一次性生成 cases_v1.json —— 之后用例正文永不改变。

规则：
- 正文文本在此固化，写入 JSON 后**任何脚本都不再生成随机 prompt**。
- 唯一允许变化的是 prompt 开头的 nonce 标记（如 "[run:0031a]"），
  作用是让前缀缓存必然失效；它只有约 8 个 token，正文长度不变。
- 改了正文 = 必须新建 cases_v2.json，旧结果不可混用。

正文生成算法与 bench_v1 原实现保持一致（random.Random(100000) 汉字散列），
只是把「随机」变成了「生成一次、然后固化」。
"""
import hashlib
import json
import random

Q_LONG = "\n\n请详细解释这段内容可能描述的技术原理，尽量展开，不少于三百字。"
Q_SHORT = "\n\n请用一句话概括上面的内容。"


def make_body(seed, words):
    rng = random.Random(seed)
    out = []
    for i in range(words):
        out.append(chr(0x4E00 + rng.randrange(0x5000)))
        if i % 12 == 11:
            out.append("。")
    return "".join(out)


def main():
    cases = {
        "C1_decode": {
            "kind": "bench",
            "prompts": [make_body(100000, 200) + Q_LONG],
            "max_tokens": 400,
            "conc": 1,
            "note": "单流 decode 吞吐（短提示，长输出）",
        },
        "C2_ttft22k": {
            "kind": "bench",
            "prompts": [make_body(100002, 12000) + Q_SHORT],
            "max_tokens": 64,
            "conc": 1,
            "note": "2.2万 token 冷启动 TTFT（engram/prefill 主指标）",
        },
        "C3_conc3": {
            "kind": "bench",
            "prompts": [make_body(100003 + i, 2500) + Q_LONG for i in range(3)],
            "max_tokens": 300,
            "conc": 3,
            "note": "三路并发聚合吞吐与 TTFT 抖动（每路正文互不相同）",
        },
        "code_python": {
            "kind": "real",
            "prompts": [
                "用 Python 写一个线程安全的 LRU 缓存类，包含 get / put / delete 三个方法，"
                "带完整类型注解和 docstring，并附一段 pytest 单元测试。直接给完整代码，不要解释。"
            ],
            "max_tokens": 500,
            "conc": 1,
            "note": "代码生成：dspark 接受率最高的典型负载",
        },
        "json_struct": {
            "kind": "real",
            "prompts": [
                "把下面的需求转成 JSON 输出：三个员工，字段为 name、role、level、skills"
                "（字符串数组），level 是 1 到 5 的整数。只输出 JSON 本体，不要任何说明文字。"
            ],
            "max_tokens": 400,
            "conc": 1,
            "note": "结构化输出：格式强约束，接受率高",
        },
        "zh_tech": {
            "kind": "real",
            "prompts": [
                "解释 vLLM 的 PagedAttention 原理，以及它和连续批处理（continuous batching）"
                "的关系，要具体到 KV cache 的内存布局层面，并举一个显存利用率提升的例子。"
            ],
            "max_tokens": 500,
            "conc": 1,
            "note": "中文技术解释：真实聊天里最常见的形态",
        },
        "math_steps": {
            "kind": "real",
            "prompts": [
                "求 sum_{n=1}^{100} n^2 的闭式表达，并写出完整推导过程，每一步都要说明依据。"
            ],
            "max_tokens": 400,
            "conc": 1,
            "note": "数学推导：过程高度可预测，dspark 友好",
        },
    }

    h = hashlib.sha256()
    for cid in sorted(cases):
        for p in cases[cid]["prompts"]:
            h.update(p.encode("utf-8"))
        h.update(str(cases[cid]["max_tokens"]).encode())

    doc = {
        "version": "cases_v1",
        "created": "2026-11-14",
        "sha256": h.hexdigest(),
        "nonce_rule": "prompt = '[' + nonce + ']\n' + body；nonce 每次运行唯一，约 8 token",
        "changelog": "初版：bench_v1 的三档正文固化为常量 + real_v1 四档真实负载用例",
        "cases": cases,
    }
    print(json.dumps(doc, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
