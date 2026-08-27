"""
chat.py
-------
Optional, EXPLAINABLE chat layer powered by Google Gemini (free tier).

Design principle: the deterministic engine computes all the numbers; the LLM
only NARRATES and ANSWERS QUESTIONS about those numbers. It is given the exact
computed facts and instructed to answer strictly from them - never to invent
scores, roles, or recommendations. This preserves the "explainable, not a black
box" thesis while letting users ask natural-language follow-ups.

Never crashes the app:
  - If the google-generativeai package isn't installed, or no API key is in
    Streamlit Secrets, gemini_available() returns False and the UI hides/greys
    the chat. The rest of the app is unaffected.

Configuration (Streamlit Cloud -> App -> Settings -> Secrets):

    [gemini]
    api_key = "your-google-ai-studio-key"
    model = "gemini-2.5-flash-lite"   # optional; sensible default used otherwise

Dependency (requirements.txt): google-generativeai
"""

from __future__ import annotations

import json
from typing import Any

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore


# ---------------------------------------------------------------------------
# Availability / config
# ---------------------------------------------------------------------------

def _api_key() -> str | None:
    if st is None:
        return None
    try:
        if "gemini" in st.secrets and st.secrets["gemini"].get("api_key"):
            return st.secrets["gemini"]["api_key"]
    except Exception:
        return None
    return None


def _configured_model() -> str | None:
    """A model name explicitly set in secrets, if any (overrides auto-detect)."""
    try:
        m = st.secrets["gemini"].get("model")  # type: ignore
        return m or None
    except Exception:
        return None


# Preference order when auto-detecting; we pick the first that the key supports.
_MODEL_PREFERENCES = (
    "flash-lite",   # cheapest / highest free limits
    "flash",
    "pro",
)


def _pick_model(genai) -> str:
    """
    Choose a model the API key can actually use for generateContent. Avoids the
    404 you get from hard-coding a name Google has retired for your key.
    """
    # explicit override wins if provided
    override = _configured_model()
    if override:
        return override if override.startswith("models/") else f"models/{override}"

    # otherwise discover from the API
    usable = []
    try:
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                usable.append(m.name)  # e.g. "models/gemini-2.5-flash"
    except Exception:
        usable = []

    if not usable:
        # last-resort guess; may still 404 but keeps a single code path
        return "models/gemini-flash-latest"

    # prefer flash-lite > flash > pro, and newer versions (sort desc by name)
    for pref in _MODEL_PREFERENCES:
        matches = sorted([n for n in usable if pref in n], reverse=True)
        if matches:
            return matches[0]
    return sorted(usable, reverse=True)[0]


def gemini_available() -> bool:
    """True only if the package is importable AND a key is configured."""
    if _api_key() is None:
        return False
    try:
        import google.generativeai  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Facts bundle - the deterministic model's output, serialized for the LLM
# ---------------------------------------------------------------------------

def build_facts(result: dict[str, Any], recs: list[dict[str, Any]],
                merged: dict[str, Any] | None,
                seniority: dict[str, Any] | None) -> dict[str, Any]:
    """
    Collect ONLY the computed facts the chat is allowed to talk about. Kept
    compact so it fits comfortably in the prompt.
    """
    ai = result.get("ai_transformation", {})
    facts: dict[str, Any] = {
        "current_role_match": result.get("current_role_match"),
        "ai_transformation_score": ai.get("score"),
        "ai_transformation_framing": ai.get("framing"),
        "current_role_tasks": [
            {"task": t["task"], "ai_impact": t["impact_label"], "exposure": t["exposure"]}
            for t in ai.get("tasks", [])
        ],
        "skill_profile": {
            bucket: [f"{e['label']} ({e['level']})" for e in result.get("skill_profile", {}).get(bucket, [])]
            for bucket in ("strong", "moderate", "emerging")
        },
        "recommended_roles": [
            {
                "role": r["title"],
                "transition_score": r["transition_score"],
                "future_fit": r["future_fit"],
                "remote_fit": r["remote_fit"],
                "estimated_time": r["estimated_time"],
                "skills_to_build": [n["label"] for n in r["skill_gap"]["need"][:6]],
                "transition_formula": r["transition_detail"]["explanation"].get("formula"),
                "future_fit_formula": r["future_fit_detail"]["explanation"].get("formula"),
            }
            for r in recs
        ],
    }
    if seniority:
        facts["role_ai_shift"] = {
            "pattern": seniority.get("pattern"),
            "headline": seniority.get("headline"),
            "production_exposure_pct": seniority.get("production_exposure"),
            "judgment_exposure_pct": seniority.get("judgment_exposure"),
        }
    if merged:
        facts["merged_future_role"] = {
            "title": merged.get("merged_title"),
            "from_role": merged.get("from_role"),
            "merges_with": merged.get("partner_role"),
            "why": merged.get("rationale"),
            "skills_to_learn": merged.get("skills_to_learn"),
            "transition_score": merged.get("transition_score"),
            "estimated_time": merged.get("estimated_time"),
        }
    return facts


# ---------------------------------------------------------------------------
# The grounded chat call
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are the assistant for 'Data Career Navigator', a career tool for data & AI "
    "professionals. A deterministic engine has ALREADY computed the results below. "
    "Your job is to explain and discuss THOSE results in a warm, concise, practical way.\n\n"
    "STRICT RULES:\n"
    "1. Use ONLY the facts provided. Do NOT invent scores, roles, skills, salaries, or timelines.\n"
    "2. If the user asks something the facts don't cover, say you can only speak to their "
    "computed results and suggest what they could adjust in the tool.\n"
    "3. When you cite a number, use the exact value from the facts.\n"
    "4. Frame everything as career evolution and opportunity.\n"
    "5. Keep answers under ~180 words unless asked for more. Use plain language.\n"
)


def ask_gemini(question: str, facts: dict[str, Any],
               history: list[dict[str, str]] | None = None) -> str:
    """
    Answer a user question grounded in the computed facts. Returns a friendly
    message string. Never raises - returns an explanatory message on failure.
    """
    if not gemini_available():
        return ("Chat is not enabled. Add a Gemini API key in the app's Secrets "
                "to turn on the assistant. Your computed results above are complete on their own.")
    try:
        import google.generativeai as genai
        genai.configure(api_key=_api_key())
        # resolve a usable model once per session and cache it
        model_name = None
        if st is not None:
            model_name = st.session_state.get("_gemini_model")
        if not model_name:
            model_name = _pick_model(genai)
            if st is not None:
                st.session_state["_gemini_model"] = model_name
        model = genai.GenerativeModel(
            model_name,
            system_instruction=_SYSTEM + "\n\nCOMPUTED FACTS (JSON):\n" + json.dumps(facts, indent=2),
        )
        convo = []
        for turn in (history or [])[-6:]:  # keep last few turns for context
            role = "user" if turn["role"] == "user" else "model"
            convo.append({"role": role, "parts": [turn["content"]]})
        convo.append({"role": "user", "parts": [question]})
        resp = model.generate_content(convo)
        return (resp.text or "").strip() or "I couldn't generate a response - please try rephrasing."
    except Exception as exc:  # keep the app alive on any API error
        return (f"The assistant is temporarily unavailable ({type(exc).__name__}). "
                "Your computed results above are unaffected.")
