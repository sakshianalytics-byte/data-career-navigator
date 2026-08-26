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

IMPACT_LABEL = {
    "automated": "🔴 High - Automated",
    "ai_assisted": "🟠 High - AI-assisted",
    "ai_augmented": "🟡 Medium - AI-augmented",
    "human_critical": "🟢 Low - Stays human",
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
    st.subheader("1. Describe your profile")

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
        top_n = st.slider("How many recommendations?", 3, 8, 5)

    st.markdown("**Rate your skills (0-100).** Leave a skill at 0 if it is emerging / not yet developed.")

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
            skills[sid] = st.slider(labels[sid], 0, 100, int(defaults.get(sid, 0)), key=f"sk_{sid}")

    run = st.button("Analyze my career path", type="primary")

    if run:
        profile = {
            "title": title, "years_experience": years, "skills": skills,
            "industry": industry, "location": location,
            "remote_preference": remote_pref,
        }
        result = analyze_profile(profile, top_n=top_n)
        # Privacy-safe: log only the matched role + skill count, never user text.
        active_skills = {k: v for k, v in skills.items() if v > 0}
        log_event("profile_analyzed",
                  f"match={result['current_role_match']}; skills={len(active_skills)}")

        # --- Current profile ---
        st.subheader("2. Where you are today")
        st.write(f"Closest role match: **{result['current_role_match']}**")
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
        st.subheader("3. How AI is transforming your current role")
        ai = result["ai_transformation"]
        st.metric("Task transformation score", f"{ai['score']}/100")
        st.caption(ai["framing"])
        df_tasks = pd.DataFrame([
            {"Task": t["task"], "AI impact": IMPACT_LABEL[t["impact"]],
             "Exposure": t["exposure"], "What happens": t["what_happens"]}
            for t in ai["tasks"]
        ])
        st.dataframe(df_tasks, use_container_width=True, hide_index=True)

        # --- Recommendations ---
        st.subheader("4. Your top recommended next roles")
        recs = result["recommendations"]
        df_recs = pd.DataFrame([
            {"Role": r["title"], "Transition": r["transition_score"],
             "Future Fit": r["future_fit"], "Remote Fit": r["remote_fit"],
             "Skill shortfall": r["transition_detail"]["skill_shortfall"],
             "Est. time": r["estimated_time"]}
            for r in recs
        ])
        st.dataframe(df_recs, use_container_width=True, hide_index=True)

        # store recs so the deep-dive selectbox works across reruns
        st.session_state["recs"] = recs

    # --- Deep dive per role (renders if we have recommendations) ---
    if st.session_state.get("recs"):
        recs = st.session_state["recs"]
        st.subheader("5. Deep dive & why")
        role_titles = [r["title"] for r in recs]
        chosen = st.selectbox("Choose a role to explore", role_titles, key="deepdive")
        r = next(x for x in recs if x["title"] == chosen)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transition", f"{r['transition_score']}/100")
        m2.metric("Future Fit", f"{r['future_fit']}/100")
        m3.metric("Remote Fit", f"{r['remote_fit']}/100")
        m4.metric("Est. time", r["estimated_time"])
        st.caption(f"O*NET occupation (grounding): {r['onet_code']} · {r['remote_note']}")

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
        st.info(r["portfolio_project"])

        with st.expander("Why am I being recommended this role? (full calculation)"):
            log_event("explain_viewed", detail=r["title"])
            st.markdown("**Transition score**")
            st.json(r["transition_detail"]["explanation"])
            st.markdown("**Future fit**")
            st.json(r["future_fit_detail"]["explanation"])
            st.markdown("**Market signals**")
            st.json(r["market"])


# ===========================================================================
# TAB 2: Job Description Analyzer
# ===========================================================================
with tab_jd:
    st.subheader("Analyze a job description")
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
        # Privacy-safe: log only the takeover % and task count, never the JD text.
        log_event("jd_analyzed", f"takeover={jd['ai_takeover_pct']}; tasks={jd['tasks_detected']}")
        if jd["tasks_detected"] == 0:
            st.warning(jd["summary"])
        else:
            st.metric("AI takeover", f"{jd['ai_takeover_pct']}%")
            st.caption(jd["summary"])

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
    "Privacy: anonymous usage events (e.g. 'analysis run') may be logged to improve the app. "
    "Your typed inputs and pasted job descriptions are never stored."
)
