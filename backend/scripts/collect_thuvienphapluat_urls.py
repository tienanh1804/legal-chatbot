#!/usr/bin/env python3
"""
Collect a limited set of document URLs from thuvienphapluat.vn based on sitemap.

Goal:
- You don't need to manually provide URLs.
- You can "cover everything" in a broad sense by collecting from multiple
  document types, but with a cap per type (so it's not too large).

This script:
1) Downloads https://thuvienphapluat.vn/SiteMap.aspx
2) Extracts all href links
3) Filters links matching known document sections:
   - /van-ban/
   - /cong-van/
   - /nghi-dinh/
   - /thong-tu/
   - /quyet-dinh/
4) Keeps at most N URLs per type (and optionally a global total cap)
5) Writes a text file: one URL per line
"""

from __future__ import annotations

import argparse
import os
import re
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Dict, List, Optional


DEFAULT_SITEMAP_URL = "https://thuvienphapluat.vn/SiteMap.aspx"


class _HrefExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k.lower() == "href" and v:
                self.hrefs.append(v)


def _fetch_html(url: str, timeout_sec: int = 30) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; LegalRAGBot/1.0; +https://example.com/bot)",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="ignore")


def _classify(url: str) -> Optional[str]:
    """
    Return one of: van-ban, cong-van, nghi-dinh, thong-tu, quyet-dinh
    """
    u = url.lower()

    # Many LawDocument pages are under "/van-ban/" but the filename slug
    # still contains the type, e.g. "/van-ban/.../nghi-dinh-...-...aspx".
    if "cong-van-" in u:
        return "cong-van"
    if "nghi-dinh-" in u:
        return "nghi-dinh"
    if "thong-tu-" in u:
        return "thong-tu"
    if "quyet-dinh-" in u:
        return "quyet-dinh"

    # Fallbacks: directory-based detection
    if "/cong-van/" in u:
        return "cong-van"
    if "/nghi-dinh/" in u:
        return "nghi-dinh"
    if "/thong-tu/" in u:
        return "thong-tu"
    if "/quyet-dinh/" in u:
        return "quyet-dinh"

    # If it's any other /van-ban/ page, keep it under van-ban bucket.
    if "/van-ban/" in u:
        return "van-ban"

    # Last-resort heuristic: sitemap may not organize links strictly by
    # path segments. Most document pages are .aspx, so include them.
    if ".aspx" in u and "sitemap" not in u and "site-map" not in u:
        return "van-ban"

    return None


def _normalize_href(href: str) -> Optional[str]:
    """
    Keep only absolute URLs under thuvienphapluat.vn.
    """
    h = href.strip()
    if not h:
        return None
    if h.startswith("//"):
        return "https:" + h
    if h.startswith("http://") or h.startswith("https://"):
        if "thuvienphapluat.vn" in h.lower():
            return h
        return None
    # Relative links: /van-ban/..., /cong-van/... etc
    if h.startswith("/"):
        joined = urllib.parse.urljoin("https://thuvienphapluat.vn", h)
        return joined

    # Ignore other relative formats
    return None


def collect_urls(
    sitemap_url: str,
    per_type_limit: int,
    total_limit: Optional[int],
) -> List[str]:
    html = _fetch_html(sitemap_url)
    parser = _HrefExtractor()
    parser.feed(html)

    buckets: Dict[str, List[str]] = {
        "van-ban": [],
        "cong-van": [],
        "nghi-dinh": [],
        "thong-tu": [],
        "quyet-dinh": [],
    }
    seen: set[str] = set()

    for raw_href in parser.hrefs:
        url = _normalize_href(raw_href)
        if not url:
            continue
        doc_type = _classify(url)
        if not doc_type:
            continue
        if url in seen:
            continue
        if len(buckets[doc_type]) >= per_type_limit:
            continue
        buckets[doc_type].append(url)
        seen.add(url)

    urls: List[str] = []
    # Keep stable order: add buckets in type order.
    for doc_type in ["van-ban", "cong-van", "nghi-dinh", "thong-tu", "quyet-dinh"]:
        urls.extend(buckets[doc_type])
        if total_limit is not None and len(urls) >= total_limit:
            return urls[:total_limit]

    if total_limit is not None:
        return urls[:total_limit]
    return urls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sitemap-url",
        default=DEFAULT_SITEMAP_URL,
        help="Sitemap URL to extract document links.",
    )
    ap.add_argument("--per-type", type=int, default=30)
    ap.add_argument("--total-limit", type=int, default=None)
    ap.add_argument(
        "--output",
        default=os.path.join("json_data", "thuvienphapluat_urls.txt"),
        help="Output file path (one URL per line).",
    )
    ap.add_argument("--sleep-ms", type=int, default=200)
    args = ap.parse_args()

    urls = collect_urls(
        sitemap_url=args.sitemap_url,
        per_type_limit=args.per_type,
        total_limit=args.total_limit,
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.output)
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")

    print(f"Collected {len(urls)} URLs -> {out_path}")


if __name__ == "__main__":
    main()

