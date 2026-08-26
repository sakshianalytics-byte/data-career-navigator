"""
app.py - Data Career Navigator (Streamlit web app)
--------------------------------------------------
Two-tab UI on the deterministic engine. Runs fully offline - no API key.

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
from src.engine import analyze_profile, recommend_transitions
from src.jd_analyzer import analyze_jd
from src.analytics import log_event
from src.role_evolution import analyze_role_evolution

st.set_page_config(
    page_title="Data Career Navigator",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design system - global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

      /* page width + breathing room */
      .block-container { max-width: 1080px; padding-top: 2rem; padding-bottom: 4rem; }

      /* hero */
      .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #9333ea 100%);
        border-radius: 20px; padding: 34px 40px; color: #fff; margin-bottom: 8px;
        box-shadow: 0 12px 30px rgba(79,70,229,.28);
      }
      .hero h1 { font-size: 30px; font-weight: 800; margin: 0 0 6px; letter-spacing: -.3px; }
      .hero p  { font-size: 15px; margin: 0; color: rgba(255,255,255,.9); max-width: 720px; line-height: 1.55; }
      .hero .chips { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }
      .hero .chip {
        background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);
        padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 500;
      }

      /* section headline */
      .sec {
        display: flex; align-items: center; gap: 10px; margin: 30px 0 6px;
        font-size: 19px; font-weight: 700; color: #1e2233;
      }
      .sec::before {
        content: ""; width: 5px; height: 22px; border-radius: 3px;
        background: linear-gradient(180deg,#4f46e5,#9333ea);
      }
      .sec-sub { color: #6b7280; font-size: 13px; margin: 0 0 14px 15px; }

      /* skill-bucket cards */
      .bucket {
        background: #fff; border: 1px solid #eceef5; border-radius: 14px; padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(16,24,64,.05); height: 100%;
      }
      .bucket h4 { margin: 0 0 10px; font-size: 14px; font-weight: 700; }
      .bucket .row { display: flex; justify-content: space-between; font-size: 13px; padding: 3px 0; color: #374151; }
      .bucket .row b { color: #4f46e5; font-variant-numeric: tabular-nums; }
      .b-strong h4 { color: #059669; } .b-mod h4 { color: #d97706; } .b-emerg h4 { color: #6b7280; }

      /* metric cards */
      div[data-testid="stMetric"] {
        background: #fff; border: 1px solid #eceef5; border-radius: 14px;
        padding: 14px 18px; box-shadow: 0 1px 3px rgba(16,24,64,.05);
      }
      div[data-testid="stMetricLabel"] p { font-size: 13px; color: #6b7280; font-weight: 600; }
      div[data-testid="stMetricValue"] { font-size: 26px; font-weight: 800; color: #1e2233; }

      /* buttons */
      .stButton > button[kind="primary"] {
        background: linear-gradient(135deg,#4f46e5,#7c3aed); border: none; border-radius: 10px;
        font-weight: 600; padding: 8px 20px; box-shadow: 0 4px 12px rgba(79,70,229,.28);
      }
      .stButton > button[kind="primary"]:hover { filter: brightness(1.06); }

      /* tabs */
      .stTabs [data-baseweb="tab-list"] { gap: 6px; }
      .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0; padding: 8px 18px; font-weight: 600;
      }

      /* dataframes: soften borders */
      div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #eceef5; }

      /* generic info/plan card */
      .card {
        background: #fff; border: 1px solid #eceef5; border-radius: 14px; padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(16,24,64,.05);
      }
      .card h4 { margin: 0 0 8px; font-size: 13px; font-weight: 700; color: #4f46e5; text-transform: uppercase; letter-spacing: .4px; }
      .card ul { margin: 0; padding-left: 18px; } .card li { font-size: 13.5px; padding: 2px 0; color: #374151; }

      .muted { color: #8a90a6; font-size: 12.5px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sec-sub">{subtitle}</div>', unsafe_allow_html=True)


IMPACT_LABEL = {
    "automated": "🔴 High · Automated",
    "ai_assisted": "🟠 High · AI-assisted",
    "ai_augmented": "🟡 Medium · AI-augmented",
    "human_critical": "🟢 Low · Stays human",
}

DEFN = {
    "transition": "How realistic it is for you to move into this role, given how much of "
                  "your current skills carry over. Higher = an easier, more attainable move. (0-100)",
    "future_fit": "How good a destination this role is overall - blends skill fit, market "
                  "demand, growth, how AI-resilient it is, and pay. Higher = a stronger long-term bet. (0-100)",
    "remote_fit": "Share of jobs in this role offered remote. Higher = easier to find remote work. (0-100)",
    "skill_shortfall": "Roughly how many skill points you still need to close to meet this "
                       "role's requirements. Lower = you're closer.",
    "transformation": "How much of your CURRENT role's day-to-day work is being reshaped by AI. "
                      "It measures task change, not risk of losing your job. (0-100)",
    "exposure": "For a single task, the share of it AI can already do. Higher = AI handles more. (0-100)",
    "ai_impact": "How AI affects a task: Automated (AI does most), AI-assisted (AI drafts, you review), "
                 "AI-augmented (AI speeds you up, you stay in control), or Stays human (needs your judgement).",
    "est_time": "A rough estimate of how long the move could take with focused upskilling.",
    "onet": "The closest official U.S. occupation code (O*NET-SOC), grounding the role in a real occupation.",
    "takeover": "The average AI exposure across the tasks detected in a job description - a rough "
                "'how much of this role could AI take on' figure.",
}

if "opened" not in st.session_state:
    st.session_state["opened"] = True
    log_event("app_open")

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🧭 Data Career Navigator</h1>
      <p>Find your next role in the AI economy. Evidence-based, explainable career guidance -
      framed around <b>career evolution, not job replacement</b>.</p>
      <div class="chips">
        <span class="chip">Runs fully offline</span>
        <span class="chip">Every score is explainable</span>
        <span class="chip">No sign-up needed</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_nav, tab_jd = st.tabs(["  Career Navigator  ", "  Job Description Analyzer  "])


# ===========================================================================
# TAB 1: Career Navigator
# ===========================================================================
with tab_nav:
    with st.expander("What do the terms mean? (Transition, Future Fit, Exposure...)"):
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

    section("Describe your profile", "Tell us where you are today - takes about a minute.")

    labels = dl.skill_labels()
    skill_id_list = dl.skill_ids()

    col_a, col_b = st.columns(2)
    with col_a:
        title = st.text_input("Current / most recent role", "Senior Data Analyst")
        years = st.slider("Years of experience", 0, 30, 9)
        industry = st.text_input("Industry", "Banking")
    with col_b:
        location = st.text_input("Location", "India")
        remote_pref = st.checkbox("Prefer remote work", value=True)

    TOP_N = 4

    st.markdown("###### Rate your skills · score out of 100")
    st.markdown('<p class="muted">0 = none / emerging · 100 = expert. Leave a skill at 0 if you '
                'don\'t have it yet.</p>', unsafe_allow_html=True)

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

    st.write("")
    run = st.button("Analyze my career path", type="primary", use_container_width=True)

    if run:
        profile = {
            "title": title, "years_experience": years, "skills": skills,
            "industry": industry, "location": location,
            "remote_preference": remote_pref,
        }
        result = analyze_profile(profile, top_n=TOP_N,
                                 direction=st.session_state.get("direction", "ic_tech"))
        active_skills = {k: v for k, v in skills.items() if v > 0}
        profile_summary = (
            f"role={title}; years={years}; industry={industry}; "
            f"location={location}; remote={remote_pref}; skills={active_skills}"
        )
        log_event("profile_analyzed",
                  detail=f"match={result['current_role_match']}; skills={len(active_skills)}",
                  user_input=profile_summary,
                  output=f"match={result['current_role_match']}; "
                         f"ai_transformation={result['ai_transformation']['score']}")
        st.session_state["profile"] = profile
        st.session_state["result"] = result
        st.session_state.pop("recs", None)

    # --- Render results from stored state (need BOTH result and profile) ---
    if st.session_state.get("result") and st.session_state.get("profile"):
        result = st.session_state["result"]

        # --- Current profile ---
        section("Where you are today",
                f"Closest role match: {result['current_role_match']}")
        sp = result["skill_profile"]
        buckets = [
            ("b-strong", "💪 Strong", "70+", "strong"),
            ("b-mod", "🔧 Moderate", "40-69", "moderate"),
            ("b-emerg", "🌱 Emerging", "below 40", "emerging"),
        ]
        bcols = st.columns(3)
        for col, (cls, head, rng, key) in zip(bcols, buckets):
            rows = "".join(
                f'<div class="row"><span>{e["label"]}</span><b>{e["level"]}</b></div>'
                for e in sp[key]
            ) or '<div class="row muted">-</div>'
            col.markdown(
                f'<div class="bucket {cls}"><h4>{head} <span class="muted">({rng})</span></h4>{rows}</div>',
                unsafe_allow_html=True,
            )

        # --- AI transformation ---
        section("How AI is transforming your current role",
                "Task transformation, not risk of losing your job.")
        ai = result["ai_transformation"]
        mcol, _ = st.columns([1, 2])
        mcol.metric("Task transformation score", f"{ai['score']}/100", help=DEFN["transformation"])
        st.caption(ai["framing"])
        df_tasks = pd.DataFrame([
            {"Task": t["task"], "AI impact": IMPACT_LABEL[t["impact"]],
             "Exposure": t["exposure"], "What happens": t["what_happens"]}
            for t in ai["tasks"]
        ])
        st.dataframe(
            df_tasks, use_container_width=True, hide_index=True,
            column_config={"Exposure": st.column_config.ProgressColumn(
                "Exposure", help=DEFN["exposure"], min_value=0, max_value=100, format="%d")},
        )

        # --- Recommendations ---
        section("Your top recommended next roles",
                "Pick the direction you want to grow, then get tailored suggestions.")
        dcol, bcol = st.columns([3, 1])
        with dcol:
            direction = st.radio(
                "Which direction do you want to grow?",
                options=["ic_tech", "ic_nontech", "people"],
                format_func=lambda a: {
                    "ic_tech": "Individual Contributor · Tech",
                    "ic_nontech": "Individual Contributor · Non-Tech",
                    "people": "People Management",
                }[a],
                key="direction", horizontal=True,
                help="We nudge your recommendations toward roles on this track.",
            )
        with bcol:
            st.write("")
            get_recs = st.button("Get recommendations", type="primary",
                                 key="get_recs", use_container_width=True)

        stored_profile = st.session_state["profile"]  # guaranteed by the block guard

        if get_recs or "recs" not in st.session_state:
            recs = recommend_transitions(
                stored_profile, top_n=TOP_N, direction=direction)
            st.session_state["recs"] = recs
            if get_recs:
                log_event("recommendations_direction",
                          detail=f"dir={direction}; top={[r['title'] for r in recs]}")
        recs = st.session_state["recs"]

        df_recs = pd.DataFrame([
            {"Role": r["title"], "Transition": r["transition_score"],
             "Future Fit": r["future_fit"], "Remote Fit": r["remote_fit"],
             "Skill shortfall": r["transition_detail"]["skill_shortfall"],
             "Est. time": r["estimated_time"]}
            for r in recs
        ])
        st.dataframe(
            df_recs, use_container_width=True, hide_index=True,
            column_config={
                "Transition": st.column_config.ProgressColumn(
                    "Transition", help=DEFN["transition"], min_value=0, max_value=100, format="%d"),
                "Future Fit": st.column_config.ProgressColumn(
                    "Future Fit", help=DEFN["future_fit"], min_value=0, max_value=100, format="%d"),
                "Remote Fit": st.column_config.ProgressColumn(
                    "Remote Fit", help=DEFN["remote_fit"], min_value=0, max_value=100, format="%d"),
                "Skill shortfall": st.column_config.NumberColumn(
                    "Skill shortfall", help=DEFN["skill_shortfall"]),
            },
        )

        # --- Merged future role (role evolution) ---
        evo = analyze_role_evolution(stored_profile, direction)
        ss = evo["seniority_shift"]
        mg = evo["merged"]
        section("Your merged future role",
                "As AI absorbs routine work, roles merge into higher-value hybrids. "
                "Here's where yours can go on your chosen track.")

        pat_color = {"de-leveling": "#dc2626", "consolidating": "#059669", "augmenting": "#d97706"}
        st.markdown(
            f'<div class="card" style="border-left:5px solid {pat_color.get(ss["pattern"], "#4f46e5")};">'
            f'<h4 style="color:#1e2233;text-transform:none;letter-spacing:0;font-size:14px;">{ss["headline"]}</h4>'
            f'<p class="muted" style="margin:4px 0 0;">Production tasks AI can take: '
            f'<b>{ss["production_exposure"]}%</b> · Judgement core (stays human): '
            f'<b>{ss["judgment_exposure"]}%</b></p>'
            f'<p style="font-size:13.5px;color:#374151;margin:8px 0 0;">{ss["detail"]}</p>'
            "</div>",
            unsafe_allow_html=True,
        )

        if mg:
            learn = "".join(f"<li>{s}</li>" for s in mg["skills_to_learn"]) or "<li>You're well covered.</li>"
            st.markdown(
                '<div class="card" style="margin-top:12px;background:linear-gradient(135deg,#f5f3ff,#eef2ff);">'
                f'<h4>Going the {mg["direction_label"]} track</h4>'
                f'<p style="font-size:15px;font-weight:700;color:#1e2233;margin:2px 0 6px;">'
                f'{mg["from_role"]} &nbsp;→&nbsp; merges with {mg["partner_role"]} &nbsp;→&nbsp; '
                f'<span style="color:#6d28d9;">{mg["merged_title"]}</span></p>'
                f'<p style="font-size:13.5px;color:#374151;margin:0 0 10px;">{mg["rationale"]}</p>'
                f'<p class="muted" style="margin:0 0 4px;">Transition score '
                f'<b>{mg["transition_score"]}/100</b> · Est. time <b>{mg["estimated_time"]}</b></p>'
                f'<p style="font-size:13px;font-weight:600;color:#4f46e5;margin:10px 0 2px;">'
                'What to learn to get there</p>'
                f'<ul style="margin:0;padding-left:18px;font-size:13.5px;color:#374151;">{learn}</ul>'
                "</div>",
                unsafe_allow_html=True,
            )

        # --- Deep dive ---
        section("Deep dive", "Explore any recommended role in detail.")
        role_titles = [r["title"] for r in recs]
        chosen = st.selectbox("Choose a role to explore", role_titles, key="deepdive")
        r = next(x for x in recs if x["title"] == chosen)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transition", f"{r['transition_score']}/100", help=DEFN["transition"])
        m2.metric("Future Fit", f"{r['future_fit']}/100", help=DEFN["future_fit"])
        m3.metric("Remote Fit", f"{r['remote_fit']}/100", help=DEFN["remote_fit"])
        m4.metric("Est. time", r["estimated_time"], help=DEFN["est_time"])
        st.markdown(f'<p class="muted">O*NET occupation: {r["onet_code"]} · {r["remote_note"]}</p>',
                    unsafe_allow_html=True)

        gcol1, gcol2 = st.columns(2)
        have_rows = "".join(
            f'<div class="row"><span>{h["label"]}</span><b>{int(h["current"])}</b></div>'
            for h in r["skill_gap"]["have"][:10]) or '<div class="row muted">-</div>'
        gcol1.markdown(
            f'<div class="bucket b-strong"><h4>✅ You already have</h4>{have_rows}</div>',
            unsafe_allow_html=True)
        need_rows = "".join(
            f'<div class="row"><span>{n["label"]}</span><b>{int(n["current"])} → {int(n["required"])}</b></div>'
            for n in r["skill_gap"]["need"]) or '<div class="row muted">You\'re well covered.</div>'
        gcol2.markdown(
            f'<div class="bucket b-mod"><h4>📈 You need to build</h4>{need_rows}</div>',
            unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="sec-sub" style="margin-left:0;font-weight:600;color:#1e2233;">90-day transition plan</div>',
                    unsafe_allow_html=True)
        rm = r["roadmap"]
        p1, p2, p3 = st.columns(3)
        for col, key, head in [(p1, "days_1_30", "Days 1-30"), (p2, "days_31_60", "Days 31-60"), (p3, "days_61_90", "Days 61-90")]:
            items = "".join(f"<li>{f}</li>" for f in rm[key]["focus"])
            col.markdown(
                f'<div class="card"><h4>{head} · {rm[key]["theme"]}</h4><ul>{items}</ul></div>',
                unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="sec-sub" style="margin-left:0;font-weight:600;color:#1e2233;">Recommended portfolio project</div>',
                    unsafe_allow_html=True)
        st.info(r["portfolio_project"])


# ===========================================================================
# TAB 2: Job Description Analyzer
# ===========================================================================
with tab_jd:
    section("Analyze a job description",
            "Paste any job description to see an AI takeover % and what stays human.")

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
    jd_text = st.text_area("Job description", sample, height=240, label_visibility="collapsed")
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
                  user_input=jd_text, output=jd_output)
        if jd["tasks_detected"] == 0:
            st.warning(jd["summary"])
        else:
            mcol, scol = st.columns([1, 2])
            mcol.metric("AI takeover", f"{jd['ai_takeover_pct']}%", help=DEFN["takeover"])
            scol.write("")
            scol.info(jd["summary"])

            c1, c2, c3 = st.columns(3)
            takeover_rows = "".join(
                f'<div class="row"><span>{a["task"]}</span><b>{a["exposure"]}</b></div>'
                for a in jd["ai_takeover_actions"]) or '<div class="row muted">-</div>'
            c1.markdown(f'<div class="bucket b-emerg"><h4 style="color:#dc2626;">🤖 AI can take over</h4>{takeover_rows}</div>',
                        unsafe_allow_html=True)
            hybrid_rows = "".join(
                f'<div class="row"><span>{a["task"]}</span><b>{a["exposure"]}</b></div>'
                for a in jd["hybrid_actions"]) or '<div class="row muted">-</div>'
            c2.markdown(f'<div class="bucket b-mod"><h4 style="color:#d97706;">🤝 Human + AI hybrid</h4>{hybrid_rows}</div>',
                        unsafe_allow_html=True)
            human_rows = "".join(
                f'<div class="row"><span>{a["task"]}</span></div>'
                for a in jd["human_critical_actions"]) or '<div class="row muted">-</div>'
            c3.markdown(f'<div class="bucket b-strong"><h4>🧠 Stays human-critical</h4>{human_rows}</div>',
                        unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="sec-sub" style="margin-left:0;font-weight:600;color:#1e2233;">Task-level breakdown</div>',
                        unsafe_allow_html=True)
            df_jd = pd.DataFrame([
                {"Task": r["task"], "Impact": IMPACT_LABEL[r["impact"]],
                 "Exposure": r["exposure"], "Why (matched text)": r["matched_text"]}
                for r in jd["breakdown"]
            ])
            st.dataframe(
                df_jd, use_container_width=True, hide_index=True,
                column_config={"Exposure": st.column_config.ProgressColumn(
                    "Exposure", help=DEFN["exposure"], min_value=0, max_value=100, format="%d")},
            )
            with st.expander("How is this calculated?"):
                st.json(jd["explanation"])


st.divider()
st.markdown(
    '<p class="muted">Seed data is illustrative - swap in O*NET + BLS + an open job-postings '
    'dataset (same schema in <code>src/data_loader.py</code>) for authoritative numbers. MIT licensed. '
    'Anonymous usage and submitted inputs may be recorded to improve the app; no names or personal '
    'identifiers are collected.</p>',
    unsafe_allow_html=True,
)
