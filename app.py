"""
app.py - Data Career Navigator (Streamlit web app)
--------------------------------------------------
Two-tab form UI on the deterministic engine. Runs fully offline - no API key.

  Tab 1: Career Navigator - fill your profile, get scored next-role recommendations.
  Tab 2: Job Description Analyzer - paste a JD, get an AI takeover % and what stays human.

Deploy: push to GitHub, then deploy on Streamlit Community Cloud with
main file path = app.py. Optional usage analytics log to a Google Sheet via
Streamlit Secrets - see src/analytics.py (never logs user content).
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from src import data_loader as dl
from src.engine import analyze_profile
from src.jd_analyzer import analyze_jd
from src.analytics import log_event

st.set_page_config(page_title="Data Career Navigator", page_icon="🧭", layout="wide")


def section_header(text: str) -> None:
    """Render an aesthetic bar-shaped section headline (no numbering)."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #4f46e5 0%, #6366f1 60%, rgba(99,102,241,0) 100%);
            color: #ffffff; padding: 10px 18px; border-radius: 8px;
            font-size: 20px; font-weight: 700; margin: 18px 0 12px;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


IMPACT_LABEL = {
    "automated": "🔴 High - Automated",
    "ai_assisted": "🟠 High - AI-assisted",
    "ai_augmented": "🟡 Medium - AI-augmented",
    "human_critical": "🟢 Low - Stays human",
}

# Plain-English definitions. Used both as hover tooltips (help=) on metrics and
# in the glossary expanders so users understand every term.
DEFN = {
    "transition": "How realistic it is for you to move into this role, given how "
                  "much of your current skills carry over. Higher = an easier, "
                  "more attainable move. (0-100)",
    "future_fit": "How good a destination this role is overall - blends skill fit, "
                  "market demand, growth, how AI-resilient it is, and pay. "
                  "Higher = a stronger long-term bet. (0-100)",
    "remote_fit": "Share of jobs in this role that are offered remote. "
                  "Higher = easier to find remote work. (0-100)",
    "skill_shortfall": "Roughly how many skill points you still need to close to "
                       "meet this role's requirements. Lower = you're closer.",
    "transformation": "How much of your CURRENT role's day-to-day work is being "
                      "reshaped by AI. It measures task change, not risk of losing "
                      "your job. (0-100)",
    "exposure": "For a single task, the share of it that AI can already do. "
                "Higher = AI can handle more of that task. (0-100)",
    "ai_impact": "How AI affects a task: Automated (AI does most of it), "
                 "AI-assisted (AI drafts, you review), AI-augmented (AI speeds you "
                 "up, you stay in control), or Stays human (needs your judgement).",
    "est_time": "A rough estimate of how long the move could take with focused "
                "upskilling.",
    "onet": "The closest official U.S. occupation code (O*NET-SOC), shown to "
            "ground the role in a real, recognised occupation.",
    "takeover": "The average AI exposure across the tasks detected in a job "
                "description - a rough 'how much of this role could AI take on' figure.",
}

# Log one app-open event per browser session.
if "opened" not in st.session_state:
    st.session_state["opened"] = True
    log_event("app_open")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧭 Data Career Navigator")
st.caption(
    "Your career in the age of AI. Evidence-based, explainable, and framed around "
    "**career evolution, not job replacement**. Runs fully offline."
)

tab_nav, tab_jd = st.tabs(["Career Navigator", "Job Description Analyzer"])


# ===========================================================================
# TAB 1: Career Navigator
# ===========================================================================
with tab_nav:
    with st.expander("📖 What do the terms mean? (Transition, Future Fit, Exposure...)"):
        st.markdown(
            f"- **Transition score** - {DEFN['transition']}\n"
            f"- **Future Fit** - {DEFN['future_fit']}\n"
            f"- **Remote Fit** - {DEFN['remote_fit']}\n"
            f"- **Skill shortfall** - {DEFN['skill_shortfall']}\n"
            f"- **Task transformation score** - {DEFN['transformation']}\n"
            f"- **Exposure** - {DEFN['exposure']}\n"
            f"- **AI impact** - {DEFN['ai_impact']}\n"
            f"- **Est. time** - {DEFN['est_time']}\n"
            f"- **O*NET code** - {DEFN['onet']}"
        )

    section_header("Describe your profile")

    labels = dl.skill_labels()
    skill_id_list = dl.skill_ids()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        title = st.text_input("Current / most recent role", "Senior Data Analyst")
        years = st.slider("Years of experience", 0, 30, 9)
        industry = st.text_input("Industry", "Banking")
    with col_b:
        location = st.text_input("Location", "India")
        remote_pref = st.checkbox("Prefer remote work", value=True)
        top_n = st.slider("Number of next-role suggestions to show", 3, 8, 5)

    st.markdown("**Rate your skills - enter a score out of 100** (0 = none/emerging, 100 = expert). "
                "Leave a skill at 0 if you don't have it yet.")

    # sensible defaults so the demo shows something meaningful immediately
    defaults = {
        "sql": 90, "bi_reporting": 85, "data_viz": 80, "statistics": 70,
        "python": 55, "data_modeling": 55, "business_analysis": 80,
        "stakeholder_mgmt": 80, "problem_framing": 70, "domain_knowledge": 75,
        "genai_llm": 25,
    }

    skills: dict[str, int] = {}
    cols = st.columns(3)
    for i, sid in enumerate(skill_id_list):
        with cols[i % 3]:
            skills[sid] = st.number_input(
                labels[sid], min_value=0, max_value=100,
                value=int(defaults.get(sid, 0)), step=5, key=f"sk_{sid}",
            )

    run = st.button("Analyze my career path", type="primary")

    # On click: compute the result and STORE it. We render from session_state
    # below so the results persist across later interactions (e.g. the deep-dive
    # dropdown), instead of disappearing when the button is no longer "pressed".
    if run:
        profile = {
            "title": title, "years_experience": years, "skills": skills,
            "industry": industry, "location": location,
            "remote_preference": remote_pref,
        }
        result = analyze_profile(profile, top_n=top_n)
        active_skills = {k: v for k, v in skills.items() if v > 0}
        profile_summary = (
            f"role={title}; years={years}; industry={industry}; "
            f"location={location}; remote={remote_pref}; skills={active_skills}"
        )
        output_summary = (
            f"match={result['current_role_match']}; "
            f"ai_transformation={result['ai_transformation']['score']}; "
            "recommendations=["
            + " | ".join(
                f"{r['title']}(transition={r['transition_score']},fit={r['future_fit']},"
                f"remote={r['remote_fit']},time={r['estimated_time']})"
                for r in result["recommendations"]
            )
            + "]"
        )
        log_event("profile_analyzed",
                  detail=f"match={result['current_role_match']}; skills={len(active_skills)}",
                  user_input=profile_summary,
                  output=output_summary)
        st.session_state["result"] = result

    # --- Render results from stored state (persists across reruns) ---
    if st.session_state.get("result"):
        result = st.session_state["result"]

        # --- Current profile ---
        section_header("Where you are today")
        st.write(f"Closest role match: **{result['current_role_match']}**")
        st.caption("**Strong** = skills you rated 70+ · **Moderate** = 40-69 · "
                   "**Emerging** = below 40 (just starting).")
        sp = result["skill_profile"]
        c1, c2, c3 = st.columns(3)
        for col, key, head in [(c1, "strong", "💪 Strong"), (c2, "moderate", "🔧 Moderate"), (c3, "emerging", "🌱 Emerging")]:
            with col:
                st.markdown(f"**{head}**")
                for e in sp[key]:
                    st.write(f"- {e['label']} ({e['level']})")
                if not sp[key]:
                    st.write("-")

        # --- AI transformation ---
        section_header("How AI is transforming your current role")
        ai = result["ai_transformation"]
        st.metric("Task transformation score", f"{ai['score']}/100",
                  help=DEFN["transformation"])
        st.caption(ai["framing"])
        st.caption(f"**AI impact** = {DEFN['ai_impact']}  \n**Exposure** = {DEFN['exposure']}")
        df_tasks = pd.DataFrame([
            {"Task": t["task"], "AI impact": IMPACT_LABEL[t["impact"]],
             "Exposure": t["exposure"], "What happens": t["what_happens"]}
            for t in ai["tasks"]
        ])
        st.dataframe(df_tasks, use_container_width=True, hide_index=True)

        # --- Recommendations ---
        section_header("Your top recommended next roles")
        st.caption(
            f"**Transition** = {DEFN['transition']}  \n"
            f"**Future Fit** = {DEFN['future_fit']}  \n"
            f"**Remote Fit** = {DEFN['remote_fit']}  \n"
            f"**Skill shortfall** = {DEFN['skill_shortfall']}  \n"
            f"**Est. time** = {DEFN['est_time']}"
        )
        recs = result["recommendations"]
        df_recs = pd.DataFrame([
            {"Role": r["title"], "Transition": r["transition_score"],
             "Future Fit": r["future_fit"], "Remote Fit": r["remote_fit"],
             "Skill shortfall": r["transition_detail"]["skill_shortfall"],
             "Est. time": r["estimated_time"]}
            for r in recs
        ])
        st.dataframe(df_recs, use_container_width=True, hide_index=True)

        # --- Deep dive per role ---
        section_header("Deep dive")
        role_titles = [r["title"] for r in recs]
        chosen = st.selectbox("Choose a role to explore", role_titles, key="deepdive")
        r = next(x for x in recs if x["title"] == chosen)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transition", f"{r['transition_score']}/100", help=DEFN["transition"])
        m2.metric("Future Fit", f"{r['future_fit']}/100", help=DEFN["future_fit"])
        m3.metric("Remote Fit", f"{r['remote_fit']}/100", help=DEFN["remote_fit"])
        m4.metric("Est. time", r["estimated_time"], help=DEFN["est_time"])
        st.caption(f"O*NET occupation (grounding): {r['onet_code']} · {r['remote_note']}",
                   help=DEFN["onet"])
        st.caption("**You already have** = skills you already meet for this role · "
                   "**You need to build** = skills where you're short (shown as your "
                   "score → the role's requirement).")

        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.markdown("**You already have**")
            for h in r["skill_gap"]["have"][:10]:
                st.write(f"- {h['label']} ({int(h['current'])})")
        with gcol2:
            st.markdown("**You need to build**")
            for n in r["skill_gap"]["need"]:
                st.write(f"- {n['label']}: {int(n['current'])} → {int(n['required'])} (gap {n['gap']})")

        st.markdown("**90-day transition plan**")
        rm = r["roadmap"]
        p1, p2, p3 = st.columns(3)
        for col, key, head in [(p1, "days_1_30", "Days 1-30"), (p2, "days_31_60", "Days 31-60"), (p3, "days_61_90", "Days 61-90")]:
            with col:
                st.markdown(f"*{head} - {rm[key]['theme']}*")
                for f in rm[key]["focus"]:
                    st.write(f"- {f}")

        st.markdown("**Recommended portfolio project**")
        st.caption("A hands-on project idea to build and show that you can do the target role.")
        st.info(r["portfolio_project"])


# ===========================================================================
# TAB 2: Job Description Analyzer
# ===========================================================================
with tab_jd:
    section_header("Analyze a job description")
    st.caption(
        "Paste any company's job description. The analyzer estimates an **AI takeover %**, "
        "lists what AI can take over, and what remains human-critical."
    )

    sample = (
        "Senior Data Analyst - FinTech\n\n"
        "Responsibilities:\n"
        "- Build and maintain weekly and monthly reporting dashboards in Power BI\n"
        "- Write SQL queries to extract, clean and transform data from the warehouse\n"
        "- Perform exploratory analysis to investigate trends and root causes\n"
        "- Partner with finance and product stakeholders to translate data into actionable insights\n"
        "- Define requirements and scope new analytics initiatives with the leadership team\n"
        "- Mentor junior analysts and lead the reporting roadmap\n"
        "- Develop Python data pipelines and deploy them to the cloud\n"
        "- Ensure compliance with data governance and privacy policies\n"
    )
    jd_text = st.text_area("Job description", sample, height=260)
    analyze = st.button("Analyze job description", type="primary", key="jd_btn")

    if analyze:
        jd = analyze_jd(jd_text)
        jd_output = (
            f"takeover={jd['ai_takeover_pct']}%; "
            f"ai_takeover=[{', '.join(a['task'] for a in jd['ai_takeover_actions'])}]; "
            f"hybrid=[{', '.join(a['task'] for a in jd['hybrid_actions'])}]; "
            f"human_critical=[{', '.join(a['task'] for a in jd['human_critical_actions'])}]"
        )
        log_event("jd_analyzed",
                  detail=f"takeover={jd['ai_takeover_pct']}; tasks={jd['tasks_detected']}",
                  user_input=jd_text,
                  output=jd_output)
        if jd["tasks_detected"] == 0:
            st.warning(jd["summary"])
        else:
            st.metric("AI takeover", f"{jd['ai_takeover_pct']}%", help=DEFN["takeover"])
            st.caption(jd["summary"])
            st.caption(
                "**🤖 AI can take over** = tasks AI can largely do or draft · "
                "**🤝 Human + AI hybrid** = AI speeds you up but you stay in control · "
                "**🧠 Stays human-critical** = needs human judgement, trust or accountability. "
                "The number next to each task is its **exposure** (how much AI can do, 0-100)."
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**🤖 AI can take over**")
                for a in jd["ai_takeover_actions"]:
                    st.write(f"- {a['task']} ({a['exposure']})")
            with c2:
                st.markdown("**🤝 Human + AI hybrid**")
                for a in jd["hybrid_actions"]:
                    st.write(f"- {a['task']} ({a['exposure']})")
                if not jd["hybrid_actions"]:
                    st.write("-")
            with c3:
                st.markdown("**🧠 Stays human-critical**")
                for a in jd["human_critical_actions"]:
                    st.write(f"- {a['task']}")

            st.markdown("**Task-level breakdown**")
            st.caption(f"**Impact** = {DEFN['ai_impact']}  \n**Exposure** = {DEFN['exposure']}")
            df_jd = pd.DataFrame([
                {"Task": r["task"], "Impact": IMPACT_LABEL[r["impact"]],
                 "Exposure": r["exposure"], "Why (matched text)": r["matched_text"]}
                for r in jd["breakdown"]
            ])
            st.dataframe(df_jd, use_container_width=True, hide_index=True)

            with st.expander("How is this calculated?"):
                st.json(jd["explanation"])


st.divider()
st.caption(
    "Seed data is illustrative. Swap in O*NET + BLS + an open job-postings dataset "
    "(same schema in src/data_loader.py) for authoritative numbers. MIT licensed."
)
st.caption(
    "Note: to improve the app, the inputs you submit (profile fields and pasted job "
    "descriptions) and anonymous usage events are recorded. No names or personal "
    "identifiers are collected."
)
