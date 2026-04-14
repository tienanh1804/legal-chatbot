"""Load procedure templates and manage wizard state (JSON in DB)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def load_template(template_id: str) -> Dict[str, Any]:
    try:
        from core.config import PROCEDURES_TEMPLATE_DIR
    except ImportError:
        from backend.core.config import PROCEDURES_TEMPLATE_DIR

    path = os.path.join(PROCEDURES_TEMPLATE_DIR, f"{template_id}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Template not found: {template_id}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_template_ids() -> List[str]:
    try:
        from core.config import PROCEDURES_TEMPLATE_DIR
    except ImportError:
        from backend.core.config import PROCEDURES_TEMPLATE_DIR

    if not os.path.isdir(PROCEDURES_TEMPLATE_DIR):
        return []
    out: List[str] = []
    for name in os.listdir(PROCEDURES_TEMPLATE_DIR):
        if name.endswith(".json"):
            out.append(name[:-5])
    return sorted(out)


def initial_state(template: Dict[str, Any]) -> Dict[str, Any]:
    fields = template.get("fields") or []
    return {
        "collected": {},
        "step_index": 0,
        "phase": "collecting",
        "total_fields": len(fields),
    }


def merge_state(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return initial_state({"fields": []})


def next_prompt(
    template: Dict[str, Any], state: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str], bool]:
    """
    Return (new_state, question_or_none, is_complete).
    If complete, question is None.
    """
    fields: List[Dict[str, Any]] = template.get("fields") or []
    collected: Dict[str, Any] = dict(state.get("collected") or {})
    idx = int(state.get("step_index") or 0)

    while idx < len(fields):
        key = fields[idx].get("key")
        if key and key not in collected:
            q = fields[idx].get("question") or fields[idx].get("label")
            state_out = {
                **state,
                "step_index": idx,
                "phase": "collecting",
            }
            return state_out, q, False
        idx += 1

    done = {
        **state,
        "step_index": len(fields),
        "phase": "ready",
    }
    return done, None, True


def apply_answer(
    template: Dict[str, Any], state: Dict[str, Any], message: str
) -> Dict[str, Any]:
    fields: List[Dict[str, Any]] = template.get("fields") or []
    idx = int(state.get("step_index") or 0)
    if idx >= len(fields):
        return state
    key = fields[idx].get("key")
    if not key:
        return state
    collected = dict(state.get("collected") or {})
    collected[key] = message.strip()
    return {
        **state,
        "collected": collected,
        "step_index": idx + 1,
    }


def render_output(template: Dict[str, Any], state: Dict[str, Any]) -> str:
    tpl = template.get("output_template") or ""
    collected = state.get("collected") or {}
    try:
        return tpl.format(**{k: str(v) for k, v in collected.items()})
    except KeyError as e:
        logger.warning("Missing key in template render: %s", e)
        return tpl + "\n\n" + "\n".join(f"{k}: {v}" for k, v in collected.items())
