#!/usr/bin/env python3
"""
Import documents from thuvienphapluat.vn into backend markdown_data.

Why URL-list based import?
- Full crawling the whole website is large and may violate robots/terms.
- This script takes an explicit list of URLs so you control scope and legality.

What it does:
1) Downloads each URL HTML
2) Extracts title and main content text using stdlib HTMLParser heuristics
3) Writes a markdown file: backend/markdown_data/<DocumentID>.md
4) Appends/updates backend/markdown_data/metadata.csv
5) (Optional) Run cache rebuild separately via build_search_resources.py

Input format:
- --urls-file must be a text file, one URL per line.

Example:
python scripts/import_thuvienphapluat.py ^
  --urls-file json_data/thuvien_urls.txt ^
  --limit 50 ^
  --sleep-ms 900
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterable, List, Optional, Tuple

from core.config import MARKDOWN_DIR


METADATA_CSV_PATH_DEFAULT = os.path.join(MARKDOWN_DIR, "metadata.csv")


def _read_urls(urls_file: str) -> List[str]:
    urls: List[str] = []
    with open(urls_file, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if not u or u.startswith("#"):
                continue
            urls.append(u)
    return urls


def _read_max_document_id(metadata_csv_path: str) -> int:
    if not os.path.exists(metadata_csv_path):
        return 0
    max_id = 0
    with open(metadata_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get("DocumentID", "").strip()
            if not doc_id:
                continue
            try:
                max_id = max(max_id, int(doc_id))
            except ValueError:
                continue
    return max_id


def _append_metadata_row(
    metadata_csv_path: str,
    document_id: int,
    source_url: str,
    title: str,
    number: str,
) -> None:
    file_exists = os.path.exists(metadata_csv_path)
    # Ensure folder exists
    os.makedirs(os.path.dirname(metadata_csv_path), exist_ok=True)

    with open(metadata_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["DocumentID", "Source", "Content", "Number"])
        writer.writerow([document_id, source_url, title, number])


def _write_markdown(
    markdown_dir: str,
    document_id: int,
    source_url: str,
    title: str,
    content_text: str,
) -> str:
    os.makedirs(markdown_dir, exist_ok=True)
    md_path = os.path.join(markdown_dir, f"{document_id}.md")

    normalized_title = title.strip().replace("\n", " ")
    src_line = f"*Source: {source_url}*"
    body = content_text.strip()

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"{src_line}\n\n")
        if normalized_title:
            f.write(f"# {normalized_title}\n\n")
        if body:
            f.write(body)

    return md_path


def _guess_number_from_title(title: str) -> str:
    t = title.strip()
    # Common patterns: "Quyết định 49/2012/QĐ-UBND", "Thông tư 101/2018/TT-BTC", ...
    m = re.search(r"\b(\d{1,4}\/\d{4}\/[A-ZĐÔƯĂÊÂỈƯƠÁÀẢÃẠĐ-]+)\b", t)
    if m:
        return m.group(1)
    return t[:120]


class _MainContentExtractor(HTMLParser):
    """
    Capture visible text within a target container.

    Targets:
    - id="divContentDoc"
    - class="content1"
    Fallback: if none matched, capture text of whole body (best-effort).
    """

    def __init__(self) -> None:
        super().__init__()
        self._in_target = False
        self._target_depth = 0
        self._captured: List[str] = []
        self._title: Optional[str] = None
        self._in_h1 = False
        self._buf_h1: List[str] = []
        self._seen_body = False
        self._capture_fallback_body = False
        self._body_depth = 0

        # HTML entities are decoded by html module; but HTMLParser provides raw.

    @property
    def captured_text(self) -> str:
        return " ".join([s.strip() for s in self._captured if s.strip()])

    @property
    def title(self) -> Optional[str]:
        if self._title:
            return self._title
        if self._buf_h1:
            return " ".join(self._buf_h1).strip()
        return None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attrs_dict = {k: (v if v is not None else "") for k, v in attrs}

        if tag.lower() == "h1":
            self._in_h1 = True
            self._buf_h1 = []

        if tag.lower() == "meta":
            # og:title
            prop = attrs_dict.get("property", "") or attrs_dict.get("name", "")
            if prop in ("og:title", "twitter:title"):
                c = attrs_dict.get("content")
                if c and not self._title:
                    self._title = c

        # If already inside target, track nested div depth
        if self._in_target and tag.lower() == "div":
            self._target_depth += 1

        # Detect target container (first matching container turns capture on)
        if tag.lower() == "div" and not self._in_target:
            div_id = attrs_dict.get("id", "")
            div_class = attrs_dict.get("class", "")

            is_div_content_doc = div_id == "divContentDoc"
            is_content1 = "content1" in div_class.split()

            if is_div_content_doc or is_content1:
                self._in_target = True
                self._target_depth = 1

        if tag.lower() == "body":
            self._seen_body = True
            self._capture_fallback_body = not self._in_target
            self._body_depth = 1

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "h1" and self._in_h1:
            self._in_h1 = False

        if self._in_target and t == "div":
            self._target_depth -= 1
            if self._target_depth <= 0:
                self._in_target = False

        if self._capture_fallback_body and t == "body":
            self._body_depth -= 1
            if self._body_depth <= 0:
                self._capture_fallback_body = False

    def handle_data(self, data: str) -> None:
        txt = html.unescape(data)
        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt:
            return

        if self._in_h1:
            self._buf_h1.append(txt)
            return

        if self._in_target:
            self._captured.append(txt)
            return

        if self._capture_fallback_body:
            self._captured.append(txt)


def _extract_title_and_text(html_text: str) -> Tuple[str, str]:
    parser = _MainContentExtractor()
    parser.feed(html_text)
    title = parser.title or ""
    content = parser.captured_text or ""

    # Clean up: remove repeated separators
    content = re.sub(r"\n{3,}", "\n\n", content)
    return title, content


def _fetch_url(url: str, timeout_sec: int = 30) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; LegalRAGBot/1.0; +https://example.com/bot)",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        # Let urllib detect encoding; fallback to utf-8.
        raw = resp.read()
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="ignore")


def iter_url_batches(urls: List[str], limit: Optional[int]) -> Iterable[str]:
    if limit is None:
        yield from urls
        return
    for i, u in enumerate(urls):
        if i >= limit:
            break
        yield u


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--urls-file",
        required=True,
        help="Text file: one URL per line (UTF-8).",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep-ms", type=int, default=900)
    ap.add_argument(
        "--markdown-dir",
        default=MARKDOWN_DIR,
        help="Directory for markdown files (default: backend/markdown_data).",
    )
    ap.add_argument(
        "--metadata-csv",
        default=METADATA_CSV_PATH_DEFAULT,
        help="metadata.csv path (default: backend/markdown_data/metadata.csv).",
    )
    args = ap.parse_args()

    urls = _read_urls(args.urls_file)
    if not urls:
        raise SystemExit("No URLs found in --urls-file")

    max_id = _read_max_document_id(args.metadata_csv)
    next_id = max_id + 1

    print(f"Found {len(urls)} URLs. Current max DocumentID={max_id}.")

    for idx, url in enumerate(iter_url_batches(urls, args.limit)):
        print(f"[{idx+1}] Importing: {url}")
        try:
            html_text = _fetch_url(url)
            title, content_text = _extract_title_and_text(html_text)

            if not title:
                title = url.split("/")[-1]
            if not content_text:
                print("  - Warning: empty content, skipping")
                time.sleep(args.sleep_ms / 1000)
                continue

            number = _guess_number_from_title(title)

            doc_id = next_id
            next_id += 1

            md_path = _write_markdown(
                markdown_dir=args.markdown_dir,
                document_id=doc_id,
                source_url=url,
                title=title,
                content_text=content_text,
            )
            _append_metadata_row(
                metadata_csv_path=args.metadata_csv,
                document_id=doc_id,
                source_url=url,
                title=title,
                number=number,
            )

            print(f"  - Saved {md_path}")
        except urllib.error.URLError as e:
            print(f"  - URLError: {e}. Skipping.")
        except Exception as e:
            print(f"  - Error: {e}. Skipping.")

        # Rate limit to be polite
        time.sleep(args.sleep_ms / 1000)

    print("Done importing.")


if __name__ == "__main__":
    main()

