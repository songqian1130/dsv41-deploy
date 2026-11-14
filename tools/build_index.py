#!/usr/bin/env python3
"""从 schemes/*/meta.json 生成根目录 index.html。

单一数据源是每个方案目录下的 meta.json —— 改数据只改 json，然后跑这个脚本。
不引入任何第三方依赖（不需要 yaml / jinja）。

用法：
    python3 tools/build_index.py
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEMES_DIR = ROOT / "schemes"
TEMPLATE = ROOT / "tools" / "index.template.html"
OUT = ROOT / "index.html"
MARKER = "__DATA__"

REQUIRED = ("id", "name", "status", "engine", "hardware", "params", "decode", "cost", "verdict")


def load_schemes() -> list[dict]:
    metas: list[dict] = []
    seen: set[str] = set()
    for path in sorted(SCHEMES_DIR.glob("*/meta.json")):
        with path.open(encoding="utf-8") as fh:
            meta = json.load(fh)
        missing = [k for k in REQUIRED if k not in meta]
        if missing:
            sys.exit(f"{path}: 缺字段 {', '.join(missing)}")
        if meta["id"] in seen:
            sys.exit(f"{path}: 方案 id 重复：{meta['id']}")
        seen.add(meta["id"])
        meta["_source"] = str(path.relative_to(ROOT))
        metas.append(meta)
    if not metas:
        sys.exit("schemes/*/meta.json 一个都没找到")
    metas.sort(key=lambda m: m.get("order", 999))
    return metas


def main() -> None:
    schemes = load_schemes()
    data = {
        "schemes": schemes,
        "built": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    template = TEMPLATE.read_text(encoding="utf-8")
    if MARKER not in template:
        sys.exit(f"模板里找不到占位符 {MARKER}：{TEMPLATE}")

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html = template.replace(MARKER, payload)
    OUT.write_text(html, encoding="utf-8")

    kb = len(html.encode("utf-8")) / 1024
    print(f"写出 {OUT.relative_to(ROOT)}  ·  {len(schemes)} 个方案  ·  {kb:.0f} KB")
    for m in schemes:
        mean = m["decode"].get("mean")
        print(f"  - {m['name']:<16} {m['engine']['family']:<10} {mean if mean is not None else '未测'} tok/s")


if __name__ == "__main__":
    main()
