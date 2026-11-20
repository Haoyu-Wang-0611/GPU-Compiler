#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 paper/*.pdf 抽成纯文本，落到 docs/ref/papers/。

依赖 pypdf（已装在 workspace 的 .deps/ 下；缺失时执行：
    python -m pip install --target .deps pypdf
）。抽出的文本只用于检索与精读辅助，引用仍以 PDF 原文为准。
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, ".deps"))

try:
    from pypdf import PdfReader
except ImportError:
    sys.exit("缺 pypdf：python -m pip install --target .deps pypdf")

OUT = os.path.join(ROOT, "docs", "ref", "papers")
os.makedirs(OUT, exist_ok=True)

for pdf in sorted(glob.glob(os.path.join(ROOT, "paper", "*.pdf"))):
    name = os.path.basename(pdf)[:-4]
    stem = name[2:] if name.startswith("英-") else name
    reader = PdfReader(pdf)
    body = "".join(
        f"\n===== [page {i+1}] =====\n" + (p.extract_text() or "")
        for i, p in enumerate(reader.pages)
    )
    dest = os.path.join(OUT, stem + ".txt")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"{dest}: {len(reader.pages)} pages, {len(body)} chars")
