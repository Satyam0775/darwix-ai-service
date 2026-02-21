# app/services/coach_service.py
"""
Identifies "coachable moments" in a transcript using keyword/phrase
heuristics + optional LLM classification.

Moment types detected:
  - objection        : price/contract/competitor concerns
  - buying_signal    : positive purchase intent
  - confusion        : customer uncertainty
  - commitment       : explicit next-step agreement
  - escalation       : frustrated / angry language
"""

from __future__ import annotations

import re
from typing import List, Dict, Any

from loguru import logger

# ── Rule definitions ──────────────────────────────────────────────────────────

_RULES: List[Dict[str, Any]] = [
    {
        "moment_type": "objection",
        "patterns": [
            r"\bnot sure\b",
            r"\btoo expensive\b",
            r"\bcan'?t afford\b",
            r"\bneed to think\b",
            r"\bnot convinced\b",
            r"\bprice is (high|too much)\b",
            r"\bbudget (concern|issue|problem)\b",
            r"\bcompetitor\b",
            r"\balready (have|using|work(ing)? with)\b",
        ],
        "confidence_base": 0.80,
    },
    {
        "moment_type": "buying_signal",
        "patterns": [
            r"\bwhen can (we|I) start\b",
            r"\bhow do (I|we) sign\b",
            r"\bsound[s]? good\b",
            r"\blet'?s (do|move forward|proceed)\b",
            r"\bI'?m (interested|ready)\b",
            r"\bcan you send (me|us) (the |a )?contract\b",
            r"\bsend (me|us) the (proposal|quote|invoice)\b",
        ],
        "confidence_base": 0.85,
    },
    {
        "moment_type": "confusion",
        "patterns": [
            r"\bdon'?t understand\b",
            r"\bnot (sure|clear) (what|how|why)\b",
            r"\bcan you (explain|clarify|repeat)\b",
            r"\bwhat (do you mean|does that mean)\b",
            r"\bconfus(ed|ing)\b",
        ],
        "confidence_base": 0.75,
    },
    {
        "moment_type": "commitment",
        "patterns": [
            r"\bI'?ll (do|send|call|follow up)\b",
            r"\bwe'?ll (schedule|arrange|set up)\b",
            r"\bnext step[s]?\b",
            r"\bfollow[- ]?up\b",
            r"\bmeeting (on|at|tomorrow|next week)\b",
        ],
        "confidence_base": 0.70,
    },
    {
        "moment_type": "escalation",
        "patterns": [
            r"\bunacceptable\b",
            r"\bfrustrat(ed|ing)\b",
            r"\banger|angry\b",
            r"\bthis is (ridiculous|terrible|awful)\b",
            r"\bworst (service|experience|support)\b",
            r"\bcancel(l?ing)? (my |the |our )?(account|subscription|contract)\b",
        ],
        "confidence_base": 0.88,
    },
]

_compiled_rules = [
    {**rule, "_re": [re.compile(p, re.IGNORECASE) for p in rule["patterns"]]}
    for rule in _RULES
]


# ── Public function ───────────────────────────────────────────────────────────

def detect_coachable_moments(
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Scan transcript segments and return detected coachable moments.

    Args:
        segments: list of {"speaker", "start_s", "end_s", "text"}

    Returns:
        list of {
          "moment_type", "text", "start_s", "end_s",
          "speaker", "confidence"
        }
    """
    moments = []
    seen_texts: set = set()   # avoid duplicate detections on same text

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text or text in seen_texts:
            continue

        for rule in _compiled_rules:
            for pattern in rule["_re"]:
                if pattern.search(text):
                    logger.debug(
                        f"Coachable moment [{rule['moment_type']}] "
                        f"@ {seg['start_s']}s: {text[:60]}"
                    )
                    moments.append({
                        "moment_type": rule["moment_type"],
                        "text":        text,
                        "start_s":     seg.get("start_s"),
                        "end_s":       seg.get("end_s"),
                        "speaker":     seg.get("speaker"),
                        "confidence":  rule["confidence_base"],
                    })
                    seen_texts.add(text)
                    break           # one rule match per segment is enough
            else:
                continue
            break

    logger.info(f"Detected {len(moments)} coachable moment(s).")
    return moments