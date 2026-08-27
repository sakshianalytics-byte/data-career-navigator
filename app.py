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
from src.analytics import (log_event, SHEET_NAVIGATOR, SHEET_JD, SHEET_MERGE)
from src.role_evolution import analyze_role_evolution, merge_two_roles
from src import chat as chatmod

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

      /* lean top bar (gradient black, white text, Times New Roman) */
      .topbar {
        background: linear-gradient(135deg, #0b0b0f 0%, #1c1c22 55%, #2b2b33 100%);
        border-radius: 14px; padding: 12px 22px; color: #fff;
        margin: 22px 0 16px; overflow: hidden;
        box-shadow: 0 6px 18px rgba(0,0,0,.28);
        display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
        font-family: "Times New Roman", Times, serif;
      }
      .topbar .brand { font-size: 22px; font-weight: 700; letter-spacing: .2px; white-space: nowrap; }
      .topbar .tag { font-size: 13px; color: rgba(255,255,255,.75); }

      /* trim the empty space between the tab bar and the first content */
      .stTabs [data-baseweb="tab-panel"] { padding-top: 6px; }

      /* section headline */
      .sec {
        display: flex; align-items: center; gap: 10px; margin: 14px 0 6px;
        font-size: 19px; font-weight: 700; color: #1e2233;
      }
      .sec::before {
        content: ""; width: 5px; height: 22px; border-radius: 3px;
        background: linear-gradient(180deg,#f97316,#ea580c);
      }
      .sec-sub { color: #6b7280; font-size: 13px; margin: 0 0 14px 15px; }

      /* skill-bucket cards */
      .bucket {
        background: #fff; border: 1px solid #eceef5; border-radius: 14px; padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(16,24,64,.05); height: 100%;
      }
      .bucket h4 { margin: 0 0 10px; font-size: 14px; font-weight: 700; }
      .bucket .row { display: flex; justify-content: space-between; font-size: 13px; padding: 3px 0; color: #374151; }
      .bucket .row b { color: #ea580c; font-variant-numeric: tabular-nums; }
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
        background: linear-gradient(135deg,#f97316,#ea580c); border: none; border-radius: 10px;
        font-weight: 600; padding: 8px 20px; box-shadow: 0 4px 12px rgba(234,88,12,.28);
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
      .card h4 { margin: 0 0 8px; font-size: 13px; font-weight: 700; color: #ea580c; text-transform: uppercase; letter-spacing: .4px; }
      .card ul { margin: 0; padding-left: 18px; } .card li { font-size: 13.5px; padding: 2px 0; color: #374151; }

      .muted { color: #8a90a6; font-size: 12.5px; }

      /* skill-group heading */
      .skillgroup {
        font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .5px;
        color: #ea580c; margin: 14px 0 4px; padding-bottom: 3px; border-bottom: 1px solid #eceef5;
      }
      /* compact skill text inputs so 32 skills fit tightly (no +/- steppers) */
      div[data-testid="stTextInput"] label p { font-size: 11.5px; margin-bottom: 1px; }
      div[data-testid="stTextInput"] input { padding: 3px 8px; }
      div[data-testid="stTextInput"] { margin-bottom: -4px; }

      /* hide the marker div used to flag a scored skill box */
      .scored-flag { display: none; }
      /* light-orange tint for any skill box the user has scored (>0):
         the marker div renders in the same column right after the input,
         so we style the text-input that is a sibling preceding it. */
      div[data-testid="column"]:has(> div div .scored-flag)
        div[data-testid="stTextInput"] input {
          background: #fff3e6;
          border: 1px solid #f9a24b;
          font-weight: 600;
          color: #b45309;
      }
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
    "transformation": "How much of your role's day-to-day work is being reshaped by AI. (0-100)",
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
# Lean top bar
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="topbar">
      <span class="brand">Data Career Navigator</span>
      <span class="tag">Find your next role in the AI economy - career evolution, not job replacement.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_nav, tab_jd, tab_merge, tab_about = st.tabs(
    ["  Career Navigator  ", "  Job Description Analyzer  ",
     "  Role Merge Explorer  ", "  About & Disclaimer  "])


# ===========================================================================
# TAB 1: Career Navigator
# ===========================================================================
with tab_nav:
    section("Describe your profile")

    labels = dl.skill_labels()

    # 5 profile inputs in a single row
    ci1, ci2, ci3, ci4, ci5 = st.columns(5)
    with ci1:
        title = st.text_input("Current / recent role", value="", placeholder="e.g. Senior Data Analyst")
    with ci2:
        years = st.number_input("Years of experience", min_value=0, max_value=40, value=0, step=1)
    with ci3:
        industry = st.text_input("Industry", value="", placeholder="e.g. Banking")
    with ci4:
        location = st.text_input("Location", value="", placeholder="e.g. India")
    with ci5:
        remote_pref = st.checkbox("Prefer remote", value=False)

    TOP_N = 4

    section("Rate your skills · score out of 10")

    # user-friendly skill groups (id lists) - defines display order + headings
    SKILL_GROUPS = [
        ("Data & Technical Tools", ["sql", "data_modeling", "bi_reporting", "data_viz",
                                    "python", "software_eng", "git_cicd", "cloud",
                                    "data_engineering", "dbt_semantic"]),
        ("AI & Machine Learning", ["machine_learning", "mlops", "genai_llm", "rag",
                                   "ai_agents", "prompt_eng"]),
        ("Analytics & Finance", ["statistics", "experimentation", "financial_analysis"]),
        ("Business & Product", ["business_analysis", "business_requirements", "problem_framing",
                                "product_mgmt", "strategy", "domain_knowledge", "documentation"]),
        ("People & Operations", ["stakeholder_mgmt", "people_mgmt", "operations"]),
        ("Risk & Compliance", ["data_governance", "risk_analysis", "regulatory_compliance"]),
    ]

    def _score(raw: str) -> int:
        """
        Parse a 1-10 skill box and convert to the engine's 0-100 scale (x10).
        Blank/invalid -> 0.
        """
        try:
            ten = max(0, min(10, float(raw)))
            return int(round(ten * 10))
        except (ValueError, TypeError):
            return 0

    # ids of every skill box, so the reset button knows what to clear
    _all_skill_ids = [sid for _, ids in SKILL_GROUPS for sid in ids]

    def _reset_skills() -> None:
        """Clear every skill text box (called before rerun by the Reset button)."""
        for sid in _all_skill_ids:
            st.session_state[f"sk_{sid}"] = ""

    # Reset button on top of the skill grid
    rcol, _ = st.columns([1, 5])
    with rcol:
        st.button("↺ Reset skills", key="reset_skills", on_click=_reset_skills,
                  use_container_width=True,
                  help="Clear all skill scores you've entered.")

    skills: dict[str, int] = {}
    for group_name, ids in SKILL_GROUPS:
        st.markdown(f'<div class="skillgroup">{group_name}</div>', unsafe_allow_html=True)
        gcols = st.columns(6)
        for i, sid in enumerate(ids):
            with gcols[i % 6]:
                key = f"sk_{sid}"
                raw = st.text_input(labels.get(sid, sid), value="", placeholder="0",
                                    key=key)
                score = _score(raw)
                skills[sid] = score
                # Tint boxes the user has actually scored (>0) light orange.
                if score > 0:
                    st.markdown(
                        f'<div class="scored-flag" data-k="{key}"></div>',
                        unsafe_allow_html=True,
                    )

    st.write("")
    run = st.button("Analyze my career path", type="primary", use_container_width=True)

    if run:
        active_skills = {k: v for k, v in skills.items() if v > 0}
        # Require a current role + at least 3 skills before analyzing.
        problems = []
        if not title.strip():
            problems.append("enter your current / most recent role")
        if len(active_skills) < 3:
            problems.append(f"score at least 3 skills (you've scored {len(active_skills)})")
        if problems:
            st.warning("Please " + " and ".join(problems) + " to get your results.")
            # clear any previous results so nothing stale shows
            st.session_state.pop("result", None)
            st.session_state.pop("profile", None)
            st.session_state.pop("recs", None)
            st.stop()

        profile = {
            "title": title, "years_experience": years, "skills": skills,
            "industry": industry, "location": location,
            "remote_preference": remote_pref,
        }
        result = analyze_profile(profile, top_n=TOP_N,
                                 direction=st.session_state.get("direction", "ic_tech"))
        profile_summary = (
            f"role={title}; years={years}; industry={industry}; "
            f"location={location}; remote={remote_pref}; skills={active_skills}"
        )
        log_event("profile_analyzed", sheet=SHEET_NAVIGATOR,
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
            ("b-mod", "🔧 Moderate", "1-69", "moderate"),
        ]
        bcols = st.columns(2)
        for col, (cls, head, rng, key) in zip(bcols, buckets):
            rows = "".join(
                f'<div class="row"><span>{e["label"]}</span><b>{e["level"]}</b></div>'
                for e in sp[key]
            ) or '<div class="row muted">-</div>'
            col.markdown(
                f'<div class="bucket {cls}"><h4>{head} <span class="muted">({rng})</span></h4>{rows}</div>',
                unsafe_allow_html=True,
            )

        # --- AI transformation --- (collapsed by default)
        with st.expander("How AI is transforming your current role", expanded=False):
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

        stored_profile = st.session_state["profile"]  # guaranteed by the block guard

        # --- Recommendations --- (collapsed by default)
        with st.expander("Your top recommended next roles", expanded=False):
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

            if get_recs or "recs" not in st.session_state:
                recs = recommend_transitions(
                    stored_profile, top_n=TOP_N, direction=direction)
                st.session_state["recs"] = recs
                if get_recs:
                    log_event("recommendations_direction", sheet=SHEET_NAVIGATOR,
                              detail=f"dir={direction}",
                              output=f"top={[r['title'] for r in recs]}")
            recs = st.session_state["recs"]

            df_recs = pd.DataFrame([
                {"Role": r["title"], "Transition": r["transition_score"],
                 "Future Fit": r["future_fit"],
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
                    "Skill shortfall": st.column_config.NumberColumn(
                        "Skill shortfall", help=DEFN["skill_shortfall"]),
                },
            )

        # recompute direction/recs outside the expander in case it was collapsed
        direction = st.session_state.get("direction", "ic_tech")
        recs = st.session_state.get("recs")
        if recs is None:
            recs = recommend_transitions(stored_profile, top_n=TOP_N, direction=direction)
            st.session_state["recs"] = recs

        # --- Merged future role (role evolution) --- (collapsed by default)
        evo = analyze_role_evolution(stored_profile, direction)
        ss = evo["seniority_shift"]
        mg = evo["merged"]
        with st.expander("Your merged future role", expanded=False):
            # Seniority-shift headline (short, as regular text)
            st.markdown(f"**{ss['headline']}**")

            if mg:
                st.markdown(f"**Going the {mg['direction_label']} track**")
                st.markdown(
                    f"{mg['from_role']} → merges with {mg['partner_role']} → "
                    f"**{mg['merged_title']}**"
                )
                st.markdown(mg["rationale"])

        # --- Deep dive --- (collapsed by default)
        with st.expander("Explore any recommended role in detail", expanded=False):
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

        # --- Ask the assistant (optional Gemini chat, grounded in the results) ---
        # Hidden from users for now. Set SHOW_CHAT = True to re-enable.
        SHOW_CHAT = False
        if SHOW_CHAT:
            section("Ask about your results",
                    "Chat with an assistant that knows your computed results above. "
                    "It explains the numbers - it can't invent new ones.")
            if not chatmod.gemini_available():
                st.caption("💬 The chat assistant is off. Add a Gemini API key in the app's "
                           "Secrets ([gemini] api_key) to enable it. Everything above works without it.")
            else:
                facts = chatmod.build_facts(result, recs, mg, ss)
                if "chat_history" not in st.session_state:
                    st.session_state["chat_history"] = []
                for turn in st.session_state["chat_history"]:
                    with st.chat_message(turn["role"],
                                         avatar="🧭" if turn["role"] == "assistant" else "🧑"):
                        st.markdown(turn["content"])
                q = st.chat_input("e.g. Why is that my best move? What should I learn first?")
                if q:
                    st.session_state["chat_history"].append({"role": "user", "content": q})
                    with st.chat_message("user", avatar="🧑"):
                        st.markdown(q)
                    with st.chat_message("assistant", avatar="🧭"):
                        with st.spinner("Thinking..."):
                            ans = chatmod.ask_gemini(q, facts, st.session_state["chat_history"])
                        st.markdown(ans)
                    st.session_state["chat_history"].append({"role": "assistant", "content": ans})
                    log_event("chat_message", sheet=SHEET_NAVIGATOR,
                              detail=f"match={result['current_role_match']}",
                              user_input=q[:500], output=ans[:1000])
                st.caption("Note: chat sends your computed results to Google Gemini. On the free "
                           "tier, Google may use inputs to improve their products.")


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
                  user_input=jd_text, output=jd_output, sheet=SHEET_JD)
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


# ===========================================================================
# TAB 3: Role Merge Explorer
# ===========================================================================
with tab_merge:
    section("Role Merge Explorer",
            "Pick any two roles to see how AI could merge them into one new role - "
            "and what skills that merged role would need.")

    role_map = {r["title"]: r["id"] for r in dl.load_roles()["roles"]}
    role_titles = sorted(role_map)

    mc1, mc2 = st.columns(2)
    with mc1:
        role_a_title = st.selectbox("First role", role_titles,
                                    index=role_titles.index("Business Analyst")
                                    if "Business Analyst" in role_titles else 0,
                                    key="merge_a")
    with mc2:
        role_b_title = st.selectbox("Second role", role_titles,
                                    index=role_titles.index("Product Manager")
                                    if "Product Manager" in role_titles else 1,
                                    key="merge_b")

    do_merge = st.button("Merge these roles", type="primary", key="merge_btn")

    if do_merge:
        if role_a_title == role_b_title:
            st.warning("Pick two different roles to merge.")
        else:
            res = merge_two_roles(role_map[role_a_title], role_map[role_b_title])
            log_event("roles_merged",
                      detail=f"{role_a_title} + {role_b_title}",
                      user_input=f"{role_a_title} + {role_b_title}",
                      output=f"merged={res['merged']['title']}; exposure={res['merged']['ai_exposure']}%",
                      sheet=SHEET_MERGE)

            # --- the two source roles side by side ---
            def role_card(side: dict) -> str:
                rows = "".join(
                    f'<div class="row"><span>{t["task"]}</span>'
                    f'<b>{t["exposure"]}%</b></div>' for t in side["tasks"])
                if not rows:
                    rows = '<div class="row muted">-</div>'
                return (f'<div class="bucket"><h4 style="color:#1e2233;">{side["title"]}'
                        f' <span class="muted">· {side["ai_exposure"]}% AI exposure</span></h4>'
                        f'{rows}</div>')

            cc1, cc2 = st.columns(2)
            cc1.markdown(role_card(res["role_a"]), unsafe_allow_html=True)
            cc2.markdown(role_card(res["role_b"]), unsafe_allow_html=True)

            # --- the merged role ---
            m = res["merged"]
            section("The merged role",
                    "AI absorbs the overlapping production work; the human-critical skills "
                    "of both roles combine into one.")

            # Gemini narrative: inherited (present) vs glue skills (to add)
            glue = set(m.get("glue_skills", []))
            inherited = [s["label"] for s in m["required_skills"] if s["label"] not in glue]
            to_add = [s["label"] for s in m["required_skills"] if s["label"] in glue] or list(glue)
            if chatmod.gemini_available():
                st.caption("✨ Generating an AI summary - this can take about 5 seconds, "
                           "please wait for it to load.")
                with st.spinner("Generating AI summary..."):
                    para = chatmod.merged_role_paragraph(
                        m["title"], res["role_a"]["title"], res["role_b"]["title"],
                        m["ai_exposure"], inherited, to_add)
            else:
                para = ""
            if para:
                st.markdown(para)

            skills_rows = "".join(
                '<div style="display:flex;justify-content:space-between;gap:12px;'
                'padding:5px 0;border-bottom:1px solid rgba(234,88,12,.15);font-size:13.5px;">'
                f'<span style="color:#374151;">{s["label"]}</span>'
                f'<b style="color:#ea580c;">{s["level"]}</b></div>'
                for s in m["required_skills"])
            st.markdown(
                '<div class="card" style="background:linear-gradient(135deg,#fff7ed,#ffedd5);">'
                f'<h4 style="margin:0 0 4px;">New role</h4>'
                f'<p style="font-size:20px;font-weight:800;color:#c2410c;margin:2px 0 8px;">{m["title"]}</p>'
                f'<p class="muted" style="margin:0 0 12px;">Overall AI exposure: '
                f'<b>{m["ai_exposure"]}%</b> · the rest stays human.</p>'
                f'<p style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;'
                f'color:#ea580c;margin:6px 0 2px;">Skills the merged role requires</p>'
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0 24px;">'
                f'{skills_rows}'
                '</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            # Mindset note - role-specific (Gemini when available, else deterministic)
            with st.spinner("Preparing the mindset note..."):
                note = chatmod.mindset_note(res["role_a"]["title"], res["role_b"]["title"])
            st.markdown(
                '<div style="border-left:4px solid #d97706;background:#fff8ef;'
                'padding:12px 16px;border-radius:0 10px 10px 0;margin-top:12px;font-size:13.5px;'
                'color:#374151;">'
                '<b style="color:#b45309;">A note on mindset:</b> '
                f'{note}</div>',
                unsafe_allow_html=True,
            )

            st.caption("Merged skills combine the human-critical skills of both roles plus "
                       "AI-orchestration skills (GenAI, prompting, agents) needed to "
                       "supervise the AI doing the routine work. Illustrative seed data.")


# ===========================================================================
# TAB 4: About & Disclaimer
# ===========================================================================
with tab_about:
    section("About this tool")
    st.markdown(
        "**Data Career Navigator** is an exploratory, evidence-based tool that models how "
        "data and AI-related roles may evolve as AI is adopted across businesses over the "
        "next 1-2 years - including how roles could merge, shift, or change in emphasis."
    )

    section("Scope & limitations")
    st.markdown(
        "- **Prototype status.** This is a prototype built on limited, illustrative data. "
        "It currently covers a limited set of data-related job profiles only.\n"
        "- **Forward-looking perspective.** It is designed to explore how the job landscape "
        "may change with AI - the merging and evolution of roles - rather than to describe "
        "the market precisely as it is today.\n"
        "- **Not career advice.** This tool is **not** a job recommendation, a hiring "
        "decision aid, or a substitute for mentorship. Please treat all results with caution "
        "and an open mind. **Do not make career decisions based solely on this tool.**\n"
        "- **Data privacy.** This tool records the inputs you submit to help improve it. "
        "Please **do not enter any personal or sensitive information** (names, contact "
        "details, employer names, or anything that could identify you)."
    )

    section("How it works, in brief")
    st.markdown(
        "Roles are represented as skill profiles and scored with transparent, deterministic "
        "math - so every number can be explained. An optional AI assistant (Google Gemini) "
        "adds written summaries but never changes the underlying scores. Underlying data is "
        "illustrative seed data and can be replaced with public sources (e.g. O*NET, BLS, "
        "open job-postings datasets) to make the numbers authoritative."
    )

    section("How roles connect & merge")
    st.markdown(
        "Roles that sit next to each other in the business process work closely and share "
        "tasks - so as AI absorbs the overlapping production work, adjacent roles become "
        "realistic candidates to **merge** into one broader role. Recommendations follow this "
        "task-flow, not just similar skills."
    )
    st.markdown(
        """
        <div style="background:#141726;border:1px solid #2a2f4a;border-radius:14px;padding:22px;color:#e7e9f3;">
          <div style="display:flex;justify-content:center;gap:16px;flex-wrap:wrap;margin-bottom:8px;">
            <div style="border:1px solid #3ba776;border-radius:10px;padding:8px 14px;font-weight:700;">DE Manager</div>
            <div style="border:1px solid #3ba776;border-radius:10px;padding:8px 14px;font-weight:700;">Analytics / ML Manager</div>
            <div style="border:1px solid #57d38c;border-radius:10px;padding:8px 14px;font-weight:700;">Business Head / Strategy / People Manager</div>
          </div>
          <div style="text-align:center;color:#f97316;font-size:20px;">&#8597;</div>
          <div style="display:flex;justify-content:center;align-items:center;gap:16px;flex-wrap:wrap;">
            <div style="border:1px solid #2a2f4a;border-radius:10px;padding:8px 14px;font-weight:700;">Data Engineer</div>
            <div style="background:linear-gradient(135deg,#f97316,#ea580c);border-radius:12px;padding:12px 18px;font-weight:800;color:#fff;">Data / Business Analyst &middot; BI Engineer<br><span style="font-weight:500;font-size:13px;opacity:.9;">the central hub</span></div>
            <div style="border:1px solid #2a2f4a;border-radius:10px;padding:8px 14px;font-weight:700;">Program / Product Manager</div>
          </div>
          <div style="text-align:center;color:#f97316;font-size:20px;">&#8595;</div>
          <div style="display:flex;justify-content:center;">
            <div style="border:1px solid #2a2f4a;border-radius:10px;padding:8px 14px;font-weight:700;">Data Scientist / Applied Scientist / ML Engineer</div>
          </div>
          <div style="text-align:center;color:#8a90a6;font-size:13px;margin-top:14px;">
            Adjacent roles (linked above) are the realistic merge candidates as AI reshapes shared tasks.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("By using this tool you acknowledge the limitations and disclaimer above.")


st.divider()
st.markdown(
    '<p class="muted">Seed data is illustrative - swap in O*NET + BLS + an open job-postings '
    'dataset (same schema in <code>src/data_loader.py</code>) for authoritative numbers. MIT licensed. '
    'Anonymous usage and submitted inputs may be recorded to improve the app; no names or personal '
    'identifiers are collected.</p>',
    unsafe_allow_html=True,
)
