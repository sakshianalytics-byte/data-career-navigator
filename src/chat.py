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
            for bucket in ("strong", "moderate")
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


# ---------------------------------------------------------------------------
# Merged-role narrative (Role Merge Explorer)
# ---------------------------------------------------------------------------

_MERGE_SYSTEM = (
    "You are a concise, practical careers writer for a data & AI career tool. "
    "Write EXACTLY ONE paragraph of about 100 words (no headings, no bullet points) "
    "describing a NEW merged role created when AI automates the overlapping routine work "
    "of two existing roles. Rules: use ONLY the facts given; do not invent skills or numbers; "
    "clearly mention the key skills the merged role INHERITS (already present from the two "
    "roles) and the NEW AI-orchestration skills the person must ADD to their kitty; frame it "
    "as an opportunity (career evolution), warm and plain-spoken."
)


# Short, role-specific "how this role thinks" descriptions. Used to ground the
# mindset note (and as the deterministic fallback when Gemini is unavailable).
THINKING_STYLES = {
    "Data Analyst": "digs deep into specific datasets to answer precise business questions",
    "Senior Data Analyst": "digs deep into data and frames the questions worth answering",
    "Business Analyst": "thinks in detail about a process and the data behind it to pin down requirements",
    "BI Lead": "thinks about consistent metrics, reporting standards, and what leaders need to see",
    "AI Analyst": "blends detailed analysis with judging what AI outputs can be trusted",
    "Analytics Engineer": "thinks in systems - clean models, pipelines, and reusable data",
    "Data Engineer": "thinks about reliability, scale, and how data flows through systems",
    "AI Data Engineer": "thinks about building dependable data + AI systems at scale",
    "Software Developer": "thinks in terms of design, edge cases, and building robust systems",
    "MLOps Engineer": "thinks about deploying, monitoring, and keeping models reliable in production",
    "Data Scientist": "thinks in hypotheses, experiments, and statistical rigour to model uncertainty",
    "ML Engineer": "thinks about turning models into performant, production-grade systems",
    "AI Engineer (GenAI)": "thinks about designing and evaluating LLM/agentic systems",
    "AI Solutions Architect": "thinks about end-to-end architecture, trade-offs, and how pieces fit",
    "Analytics Manager": "thinks about priorities, people, and turning analysis into decisions",
    "AI-Augmented Analytics Lead": "thinks about steering an AI-assisted analytics team and its direction",
    "Program Manager": "thinks broadly about coordination, timelines, risks, and cross-team buy-in",
    "Product Manager": "thinks across the whole product - users, features, and business outcomes",
    "AI Data Product Manager": "thinks about the AI product's value, roadmap, and adoption",
    "Strategy Manager": "thinks about the big picture, options, and long-term bets under uncertainty",
    "Quantitative Risk Analyst": "thinks in models, probabilities, and downside scenarios",
    "Risk & Compliance Analyst (Ops/QA)": "thinks about controls, accuracy, and adherence to policy",
    "Financial Analyst": "thinks in numbers, forecasts, and the financial impact of choices",
}


def thinking_style(role_title: str) -> str:
    return THINKING_STYLES.get(role_title, "approaches problems in its own distinct way")


_MINDSET_SYSTEM = (
    "You write a short, specific 'note on mindset' for a career tool. Given two roles and a "
    "one-line description of how each ROLE tends to think, write 2-3 sentences (about 55 words) "
    "explaining that merging their SKILLS is easy but merging their WAYS OF THINKING is the real "
    "challenge. Contrast the two thinking styles concretely and specifically for THESE two roles "
    "(do not use generic wording), then say excelling in the merged role means deliberately "
    "building a new blended mindset. Warm, plain, no headings or bullets."
)


def mindset_note(role_a: str, role_b: str) -> str:
    """
    Role-specific mindset note. Uses Gemini when available for a tailored note;
    otherwise falls back to a deterministic sentence built from THINKING_STYLES.
    """
    style_a = thinking_style(role_a)
    style_b = thinking_style(role_b)
    fallback = (
        f"This tool can merge skills, but not ways of thinking. A {role_a} {style_a}, "
        f"while a {role_b} {style_b}. The merged role is not just the sum of both skill "
        "sets - excelling in it means deliberately building a new mindset that blends "
        "both ways of thinking, and how far you succeed depends on how well you develop it."
    )
    if not gemini_available():
        return fallback
    try:
        import google.generativeai as genai
        genai.configure(api_key=_api_key())
        facts = {"role_a": role_a, "role_a_thinks": style_a,
                 "role_b": role_b, "role_b_thinks": style_b}
        model_name = st.session_state.get("_gemini_model") if st is not None else None
        if not model_name:
            model_name = _pick_model(genai)
            if st is not None:
                st.session_state["_gemini_model"] = model_name
        model = genai.GenerativeModel(
            model_name,
            system_instruction=_MINDSET_SYSTEM + "\n\nFACTS (JSON):\n" + json.dumps(facts),
        )
        resp = model.generate_content("Write the note.")
        return (resp.text or "").strip() or fallback
    except Exception:
        return fallback


def merged_role_paragraph(merged_title: str, role_a: str, role_b: str,
                          ai_exposure: int, inherited_skills: list[str],
                          add_skills: list[str]) -> str:
    """
    Generate a ~100-word paragraph describing the merged role. Returns "" if
    Gemini isn't available so the caller can just show the structured grid.
    """
    if not gemini_available():
        return ""
    try:
        import google.generativeai as genai
        genai.configure(api_key=_api_key())
        facts = {
            "merged_role": merged_title,
            "merges": [role_a, role_b],
            "overall_ai_exposure_pct": ai_exposure,
            "skills_inherited_present": inherited_skills,
            "ai_orchestration_skills_to_add": add_skills,
        }
        model_name = None
        if st is not None:
            model_name = st.session_state.get("_gemini_model")
        if not model_name:
            model_name = _pick_model(genai)
            if st is not None:
                st.session_state["_gemini_model"] = model_name
        model = genai.GenerativeModel(
            model_name,
            system_instruction=_MERGE_SYSTEM + "\n\nFACTS (JSON):\n" + json.dumps(facts, indent=2),
        )
        resp = model.generate_content("Write the paragraph.")
        return (resp.text or "").strip()
    except Exception:
        return ""
