"""
app.py - Data Career Navigator (Streamlit chat app)
----------------------------------------------------
A beautified, chat-style Streamlit app built on the same deterministic engine.
Runs fully offline - no API key required.

Conversation flow:
  - Type your profile in plain English, e.g.
      "I'm a data analyst with 9 years experience. I know SQL, Power BI, Python
       and some GenAI. I prefer remote."
    -> the bot extracts your skills and returns your next-role recommendations.
  - Paste a job description (anything with several 'Responsibilities' lines)
    -> the bot returns an AI takeover % and what stays human.

Deploy: push to GitHub, then deploy on Streamlit Community Cloud with
main file path = app.py. Usage analytics (optional) log to a Google Sheet via
Streamlit Secrets - see src/analytics.py.
"""

from __future__ import annotations

import re
import streamlit as st

from src import data_loader as dl
from src.engine import analyze_profile
from src.jd_analyzer import analyze_jd
from src.analytics import log_event

st.set_page_config(page_title="Data Career Navigator", page_icon="🧭", layout="centered")

# ---------------------------------------------------------------------------
# Styling - a cleaner, "beautified" chat look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .main { max-width: 820px; }
      .hero { text-align:center; padding: 8px 0 4px; }
      .hero h1 { margin:0; font-size: 30px; }
      .hero p { color:#6b7280; margin:6px 0 0; }
      .badge { display:inline-block; background:#eef2ff; color:#3730a3;
               padding:3px 10px; border-radius:999px; font-size:12px; margin-top:8px; }
      .stChatMessage { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🧭 Data Career Navigator</h1>
      <p>Your career in the age of AI - evidence-based, explainable, framed around
      <b>career evolution, not job replacement</b>.</p>
      <span class="badge">Runs offline · every score is explainable</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Lightweight, deterministic message parsing (no LLM)
# ---------------------------------------------------------------------------
SKILL_ALIASES = {
    "sql": ["sql"],
    "bi_reporting": ["power bi", "powerbi", "tableau", "looker", "bi", "reporting", "dashboards"],
    "data_viz": ["visualization", "visualisation", "data viz", "charts"],
    "statistics": ["statistics", "stats", "statistical"],
    "experimentation": ["a/b", "ab testing", "experimentation", "experiments"],
    "python": ["python", "pandas"],
    "software_eng": ["software engineering", "software development"],
    "git_cicd": ["git", "ci/cd", "cicd"],
    "cloud": ["aws", "azure", "gcp", "cloud"],
    "data_engineering": ["data engineering", "etl", "pipelines", "spark"],
    "dbt_semantic": ["dbt", "semantic layer", "metrics layer"],
    "machine_learning": ["machine learning", "ml ", "scikit", "modeling", "modelling"],
    "mlops": ["mlops"],
    "genai_llm": ["genai", "gen ai", "llm", "generative ai", "gpt"],
    "rag": ["rag", "vector search", "retrieval"],
    "ai_agents": ["agents", "agentic"],
    "ai_evaluation": ["evaluation", "evals", "guardrails"],
    "prompt_eng": ["prompt", "prompting"],
    "business_analysis": ["business analysis", "business analyst"],
    "stakeholder_mgmt": ["stakeholder", "stakeholders"],
    "problem_framing": ["problem framing", "scoping", "strategy"],
    "product_mgmt": ["product management", "product manager"],
    "domain_knowledge": ["banking", "finance", "healthcare", "retail", "domain"],
}


def looks_like_jd(text: str) -> bool:
    """Heuristic: multi-line, has responsibility-ish cues, and is fairly long."""
    cues = ["responsibilities", "requirements", "you will", "qualifications",
            "what you", "role:", "we are looking"]
    low = text.lower()
    has_cue = any(c in low for c in cues)
    many_lines = text.count("\n") >= 3 or low.count("- ") >= 3
    return len(text) > 220 and (has_cue or many_lines)


def extract_years(text: str) -> int:
    m = re.search(r"(\d{1,2})\s*(?:\+)?\s*(?:years|yrs|year)", text.lower())
    return int(m.group(1)) if m else 5


def extract_skills(text: str) -> dict:
    """Map mentioned skills to reasonable proficiency values (deterministic)."""
    low = text.lower()
    skills: dict[str, int] = {}
    for sid, aliases in SKILL_ALIASES.items():
        for a in aliases:
            if a in low:
                # crude proficiency heuristic from qualifier words
                if any(q in low for q in ["expert", "strong", "advanced", "lead"]):
                    skills[sid] = 85
                elif any(q in low for q in ["some", "basic", "learning", "beginner", "little"]):
                    skills[sid] = 35
                else:
                    skills[sid] = 65
                break
    return skills


def extract_profile(text: str) -> dict:
    low = text.lower()
    # crude role title = first line or a known keyword
    title = text.strip().split("\n")[0][:60] or "Data professional"
    return {
        "title": title,
        "years_experience": extract_years(text),
        "skills": extract_skills(text),
        "remote_preference": "remote" in low,
    }


# ---------------------------------------------------------------------------
# Rendering helpers (return markdown strings for chat bubbles)
# ---------------------------------------------------------------------------
IMPACT = {
    "automated": "🔴 Automated", "ai_assisted": "🟠 AI-assisted",
    "ai_augmented": "🟡 AI-augmented", "human_critical": "🟢 Human-critical",
}


def render_profile_reply(profile: dict) -> str:
    if not profile["skills"]:
        return ("I couldn't spot any skills in that. Try naming a few, e.g. "
                "*\"Data analyst, 9 years, SQL, Power BI, Python, some GenAI, prefer remote.\"*")
    result = analyze_profile(profile, top_n=4)
    # Privacy-safe: log only the matched role + skill count, never the raw text.
    log_event("profile_analyzed",
              f"match={result['current_role_match']}; skills={len(profile['skills'])}")
    ai = result["ai_transformation"]
    lines = [
        f"**Closest role match:** {result['current_role_match']}",
        f"**AI task-transformation score:** {ai['score']}/100  \n"
        f"_{ai['framing']}_",
        "",
        "**Your top next roles:**",
    ]
    for i, r in enumerate(result["recommendations"], 1):
        lines.append(
            f"{i}. **{r['title']}** — transition {r['transition_score']}/100 · "
            f"future fit {r['future_fit']}/100 · remote {r['remote_fit']}/100 · "
            f"~{r['estimated_time']}"
        )
    top = result["recommendations"][0]
    needs = ", ".join(n["label"] for n in top["skill_gap"]["need"][:6]) or "you're well covered"
    lines += [
        "",
        f"**To reach {top['title']}, build:** {needs}.",
        "",
        "**90-day plan:**",
        f"- Days 1-30 ({top['roadmap']['days_1_30']['theme']}): "
        + "; ".join(top["roadmap"]["days_1_30"]["focus"]),
        f"- Days 31-60 ({top['roadmap']['days_31_60']['theme']}): "
        + "; ".join(top["roadmap"]["days_31_60"]["focus"]),
        f"- Days 61-90 ({top['roadmap']['days_61_90']['theme']}): "
        + "; ".join(top["roadmap"]["days_61_90"]["focus"]),
        "",
        f"**Portfolio project:** {top['portfolio_project']}",
        "",
        "_Want the full \"why this role\" math? Ask \"explain\" or paste a job description to analyze it._",
    ]
    st.session_state["last_result"] = result
    return "\n".join(lines)


def render_jd_reply(text: str) -> str:
    jd = analyze_jd(text)
    # Privacy-safe: log only the takeover % and task count, never the JD text.
    log_event("jd_analyzed",
              f"takeover={jd['ai_takeover_pct']}; tasks={jd['tasks_detected']}")
    if jd["tasks_detected"] == 0:
        return jd["summary"]
    lines = [
        f"**AI takeover: {jd['ai_takeover_pct']}%** ({jd['tasks_detected']} tasks detected)",
        f"_{jd['summary']}_",
        "",
        "**🤖 AI can take over:**",
    ]
    lines += [f"- {a['task']} ({IMPACT[a['impact']]}, exposure {a['exposure']})"
              for a in jd["ai_takeover_actions"]]
    if jd["hybrid_actions"]:
        lines += ["", "**🤝 Human + AI hybrid:**"]
        lines += [f"- {a['task']} (exposure {a['exposure']})" for a in jd["hybrid_actions"]]
    lines += ["", "**🧠 Stays human-critical:**"]
    lines += [f"- {a['task']} — {a['human_value']}" for a in jd["human_critical_actions"]]
    return "\n".join(lines)


def render_explain() -> str:
    log_event("explain_requested")
    result = st.session_state.get("last_result")
    if not result:
        return "Tell me about your background first, then I can explain the scoring."
    top = result["recommendations"][0]
    td = top["transition_detail"]["explanation"]
    fd = top["future_fit_detail"]["explanation"]
    return (
        f"**Why {top['title']}?**\n\n"
        f"**Transition score** = `{td['formula']}`  \n"
        f"transferable {td['transferable_pct']}%, readiness {td['readiness']}, "
        f"experience factor {td['experience_factor']}.\n\n"
        f"**Future fit** = `{fd['formula']}`  \n"
        f"components: {fd['components']}.\n\n"
        "Every number is computed by the engine - no black box."
    )


# ---------------------------------------------------------------------------
# Chat state + loop
# ---------------------------------------------------------------------------
WELCOME = (
    "Hi! I'm your Data Career Navigator. Two things I can do:\n\n"
    "1. **Career advice** — tell me your background, e.g. "
    "*\"Senior data analyst, 9 years, SQL, Power BI, Python, some GenAI, prefer remote.\"*\n"
    "2. **Job-description analysis** — paste a JD and I'll tell you how much AI can take over.\n\n"
    "You can also type **explain** after a recommendation to see the math."
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": WELCOME}]
    log_event("app_open")  # once per new session

with st.sidebar:
    st.header("Examples")
    st.caption("Copy one into the chat:")
    st.code("Senior data analyst, 9 years, SQL,\nPower BI, Python, some GenAI,\nprefer remote", language=None)
    st.code("Data engineer, 6 yrs, strong Python,\nSQL, cloud, data engineering, dbt", language=None)
    st.divider()
    if st.button("Reset conversation"):
        st.session_state["messages"] = [{"role": "assistant", "content": WELCOME}]
        st.session_state.pop("last_result", None)
        st.rerun()
    st.caption("Seed data is illustrative. Swap in O*NET/BLS/postings for authoritative numbers.")
    st.caption("Privacy: anonymous usage events (e.g. 'analysis run') may be logged to improve "
               "the app. Your typed text and pasted job descriptions are never stored.")

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"], avatar="🧭" if msg["role"] == "assistant" else "🧑"):
        st.markdown(msg["content"])

prompt = st.chat_input("Describe your background, or paste a job description...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    low = prompt.strip().lower()
    if low in ("explain", "why", "show the math", "explain the math"):
        reply = render_explain()
    elif looks_like_jd(prompt):
        reply = render_jd_reply(prompt)
    else:
        reply = render_profile_reply(extract_profile(prompt))

    with st.chat_message("assistant", avatar="🧭"):
        st.markdown(reply)
    st.session_state["messages"].append({"role": "assistant", "content": reply})
