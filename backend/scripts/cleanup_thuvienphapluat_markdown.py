#!/usr/bin/env python3
"""
Cleanup noisy imported markdown from thuvienphapluat.vn.

Problem:
- Some URLs collected from sitemap are not actual legal documents
  (e.g. contact/help pages). Our extractor may include large JS/CSS/boilerplate
  text in the markdown content.

Solution:
- Extract the "legal core" of the markdown body by finding the earliest
  occurrence of typical Vietnamese legal markers (case-insensitive).
- Rewrite the markdown file keeping the original `*Source:` and `# Title`
  lines, but replace the body with the legal core (or empty if none found).
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Iterable, Optional, Tuple


LEGAL_MARKERS: Tuple[str, ...] = (
    # Core document markers
    r"\bQUYẾT\s+ĐỊNH\b",
    r"\bNGHỊ\s+ĐỊNH\b",
    r"\bTHÔNG\s+TƯ\b",
    # Structural markers frequently present in legal docs
    r"\bCăn\s+cứ\b",
    r"\bĐiều\s+\d+\b",
)


def _parse_markdown_sections(md_text: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Split markdown into (source_line, title_line, body).
    """
    # Normalize newlines
    text = md_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    source_line: Optional[str] = None
    title_line: Optional[str] = None
    idx = 0

    if idx < len(lines) and lines[idx].startswith("*Source:"):
        source_line = lines[idx].strip()
        idx += 1

    # Skip optional blank lines
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    if idx < len(lines) and lines[idx].startswith("# "):
        title_line = lines[idx].strip()
        idx += 1

    body = "\n".join(lines[idx:]).lstrip("\n")
    return source_line, title_line, body


def _extract_legal_core(body: str) -> str:
    """
    Return the substring of `body` starting at the earliest occurrence
    of any `LEGAL_MARKERS` (case-insensitive). If none found, return "".
    """
    if not body.strip():
        return ""

    # Strong structural requirement:
    # Legal documents almost always have sections like "Điều 1", "Điều 2", ...
    # Many non-document pages may mention "Nghị định/Thông tư" in banners,
    # so we require the "Điều <number>" pattern to avoid keeping boilerplate.
    if re.search(r"Điều\s*\d+", body, flags=re.IGNORECASE) is None:
        return ""

    # We'll search in a normalized uppercase version to make regex more stable.
    # But we still use case-insensitive regex for safety.
    lowered = body.lower()

    best_start: Optional[int] = None
    for marker in LEGAL_MARKERS:
        m = re.search(marker, body, flags=re.IGNORECASE)
        if not m:
            continue
        start = m.start()
        if best_start is None or start < best_start:
            best_start = start

    if best_start is None:
        return ""

    core = body[best_start:]
    # Basic whitespace normalization
    core = core.strip()
    core = re.sub(r"\n{3,}", "\n\n", core)
    return core


def _cleanup_one_file(path: str) -> bool:
    """
    Rewrite markdown at `path`. Returns True if file was changed.
    """
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()

    source_line, title_line, body = _parse_markdown_sections(original)
    legal_core = _extract_legal_core(body)

    # Reconstruct
    out_lines = []
    if source_line:
        out_lines.append(source_line)
        out_lines.append("")
    if title_line:
        out_lines.append(title_line)
        out_lines.append("")
    if legal_core:
        out_lines.append(legal_core)

    cleaned = "\n".join(out_lines).strip() + ("\n" if out_lines else "")

    if cleaned == original:
        return False

    with open(path, "w", encoding="utf-8") as f:
        f.write(cleaned)
    return True


def iter_markdown_files(directory: str, min_id: Optional[int], max_id: Optional[int]) -> Iterable[str]:
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".md"):
                continue
            base = file[:-3]
            if not base.isdigit():
                continue
            doc_id = int(base)
            if min_id is not None and doc_id < min_id:
                continue
            if max_id is not None and doc_id > max_id:
                continue
            yield os.path.join(root, file)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown-dir", default="backend/markdown_data")
    ap.add_argument("--min-id", type=int, default=None)
    ap.add_argument("--max-id", type=int, default=None)
    args = ap.parse_args()

    markdown_dir = args.markdown_dir
    if not os.path.isabs(markdown_dir):
        # Relative to repo root when running on host; in Docker this should already exist.
        markdown_dir = os.path.normpath(markdown_dir)

    changed = 0
    total = 0
    for path in iter_markdown_files(markdown_dir, args.min_id, args.max_id):
        total += 1
        if _cleanup_one_file(path):
            changed += 1

    print(f"Cleanup done. Total files: {total}, changed: {changed}")


if __name__ == "__main__":
    main()

