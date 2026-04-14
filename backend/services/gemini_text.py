"""Gemini helpers for summarization, extraction, and procedure prefill."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _configure():
    import google.generativeai as genai

    try:
        from core.config import GEMINI_API_KEYS, GEMINI_MODEL, GEMINI_MAX_TOKENS, GEMINI_TEMPERATURE
    except ImportError:
        from backend.core.config import (
            GEMINI_API_KEYS,
            GEMINI_MODEL,
            GEMINI_MAX_TOKENS,
            GEMINI_TEMPERATURE,
        )

    if not GEMINI_API_KEYS:
        raise RuntimeError("No GEMINI_API_KEY configured")
    genai.configure(api_key=GEMINI_API_KEYS[0])
    return genai, GEMINI_MODEL, GEMINI_MAX_TOKENS, GEMINI_TEMPERATURE


def gemini_text(prompt: str, max_tokens: Optional[int] = None) -> str:
    """Single-shot text generation."""
    genai, model_name, default_max, temperature = _configure()
    mt = max_tokens or default_max
    model = genai.GenerativeModel(model_name=model_name)
    r = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=mt,
        ),
    )
    return (r.text or "").strip()


def summarize_text(long_text: str, focus: Optional[str] = None) -> str:
    focus_line = f"Trọng tâm: {focus}\n" if focus else ""
    prompt = f"""Bạn là trợ lý pháp lý. Viết bản tóm tắt ngắn gọn, rõ ràng bằng tiếng Việt.
{focus_line}
Nội dung:
---
{long_text[:48000]}
---
Tóm tắt (có thể dùng gạch đầu dòng, không bịa thêm sự kiện không có trong văn bản):"""
    return gemini_text(prompt, max_tokens=2048)


def extract_key_info(long_text: str, instruction: str) -> str:
    prompt = f"""Dựa vào văn bản sau, thực hiện: {instruction}
Chỉ trả lời dựa trên văn bản; nếu không có thông tin, ghi "Không có trong văn bản".

Văn bản:
---
{long_text[:48000]}
---"""
    return gemini_text(prompt, max_tokens=1536)


def prefill_fields_from_text(
    blob: str, field_keys: List[str], field_labels: List[str]
) -> Dict[str, str]:
    """Ask Gemini for JSON mapping key -> extracted value."""
    pairs = "\n".join(
        f"- {k}: ({label})" for k, label in zip(field_keys, field_labels)
    )
    prompt = f"""Bạn nhận được văn bản có thể là CCCD, hộ khẩu, hoặc giấy tờ khác.
Trích các trường sau nếu tìm thấy trong văn bản. Trả về DUY NHẤT một JSON object, không markdown.

Các trường:
{pairs}

Văn bản:
---
{blob[:20000]}
---

JSON dạng {{"ho_ten": "...", "ngay_sinh": "..."}} — chỉ gồm các khóa có giá trị chắc chắn."""
    raw = gemini_text(prompt, max_tokens=1024)
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return {str(k): str(v) for k, v in data.items() if v}
    except json.JSONDecodeError:
        logger.warning("prefill JSON parse failed: %s", raw[:200])
        return {}
