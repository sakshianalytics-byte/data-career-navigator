"""
engine.py
---------
Deterministic scoring engine for the Data Career Navigator.

Design principle: career advice is a *data problem*, not a black-box answer.
Every score is computed with transparent math over skill vectors and market
signals, and every result carries an `explanation` so the UI can answer
"Why am I being recommended this role?".

No network calls, no API key required. Runs fully offline.

Main entry point
----------------
    from src.engine import analyze_profile
    result = analyze_profile(profile)

`profile` is a dict:
    {
      "title": "Senior Data Analyst",
      "years_experience": 9,
      "skills": {"sql": 90, "bi_reporting": 85, "python": 55, ...},  # 0-100
      "industry": "Banking",
      "location": "India",
      "remote_preference": True,
      "interests": ["genai_llm", "business_analysis"],  # skill ids or free text
    }
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import data_loader as dl

# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def _vector_from_map(skill_map: dict[str, float]) -> np.ndarray:
    """Turn a {skill_id: value} map into an ordered numpy vector (0-100)."""
    ids = dl.skill_ids()
    return np.array([float(skill_map.get(sid, 0.0)) for sid in ids], dtype=float)


def role_vector(role: dict[str, Any]) -> np.ndarray:
    return _vector_from_map(role.get("vector", {}))


def profile_vector(profile: dict[str, Any]) -> np.ndarray:
    return _vector_from_map(profile.get("skills", {}))


def career_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Symmetric Euclidean 'skill distance' between two roles, normalized to a
    0-100 point scale so it reads like "you are N skill points away". This is
    the human-readable distance shown in the UI.
    """
    diff = vec_a - vec_b
    rms = float(np.sqrt(np.mean(diff ** 2)))
    return round(rms, 1)


def directional_shortfall(profile_vec: np.ndarray, target_vec: np.ndarray) -> float:
    """
    How far the profile falls SHORT of the target's requirements.

    Only counts skills where the user is below what the target needs (having
    extra, irrelevant skills is not a penalty). Averaged only over the skills
    the target actually requires, and scaled by how demanding each requirement
    is. Returns a 0-100 "points to close" figure.
    """
    required = target_vec > 0
    if not np.any(required):
        return 0.0
    shortfall = np.clip(target_vec - profile_vec, 0, None)
    # weight the shortfall by requirement strength: missing a core skill hurts
    # more than missing a lightly-required one.
    weights = target_vec[required]
    gaps = shortfall[required]
    weighted = float(np.sum(gaps * weights) / np.sum(weights))
    return round(weighted, 1)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


# ---------------------------------------------------------------------------
# 1. Current profile analysis
# ---------------------------------------------------------------------------

def classify_skills(profile: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Bucket the user's skills into strong (>=70) and moderate (>0 and <70).
    Skills scored 0 are omitted entirely (not shown).
    """
    labels = dl.skill_labels()
    strong, moderate = [], []
    for sid, val in profile.get("skills", {}).items():
        if val <= 0:
            continue  # skip skills the user doesn't have
        entry = {"id": sid, "label": labels.get(sid, sid), "level": val}
        if val >= 70:
            strong.append(entry)
        else:
            moderate.append(entry)
    for bucket in (strong, moderate):
        bucket.sort(key=lambda e: e["level"], reverse=True)
    return {"strong": strong, "moderate": moderate}


def role_transformation_score(profile: dict[str, Any]) -> dict[str, Any]:
    """
    AI transformation score for the user's CURRENT role.

    We match the current role to the closest seed role, then aggregate the AI
    exposure of the tasks that role emphasises, weighted by how present each
    task's skills are in the profile. Framed as 'task transformation', not risk
    of job loss.
    """
    from .jd_analyzer import score_tasks_for_role  # local import avoids cycle

    matched = closest_role(profile)
    task_rows = score_tasks_for_role(matched["role"])
    if task_rows:
        weighted = sum(t["exposure"] * t["weight"] for t in task_rows)
        total_w = sum(t["weight"] for t in task_rows)
        score = round(weighted / total_w) if total_w else 0
    else:
        score = 0

    # Framing scales with the score so the words match the number.
    if score >= 65:
        framing = ("Your role is being heavily reshaped by AI - most day-to-day tasks are "
                   "already AI-assisted or automated.")
    elif score >= 45:
        framing = ("Your role is undergoing significant task transformation - a large share "
                   "of the work is being reshaped by AI.")
    elif score >= 30:
        framing = ("Your role is being moderately reshaped by AI - some tasks are changing "
                   "while the core stays human.")
    else:
        framing = ("Your role is only lightly touched by AI so far - most of the work still "
                   "depends on human judgement.")

    return {
        "score": score,
        "matched_role": matched["role"]["title"],
        "matched_role_id": matched["role"]["id"],
        "similarity": round(matched["similarity"], 3),
        "tasks": task_rows,
        "framing": framing,
    }


# ---------------------------------------------------------------------------
# 2. Role matching
# ---------------------------------------------------------------------------

def closest_role(profile: dict[str, Any]) -> dict[str, Any]:
    """Find the seed role whose vector is most similar to the profile."""
    p_vec = profile_vector(profile)
    best = None
    for role in dl.load_roles()["roles"]:
        sim = cosine_similarity(p_vec, role_vector(role))
        if best is None or sim > best["similarity"]:
            best = {"role": role, "similarity": sim}
    return best


# ---------------------------------------------------------------------------
# 3. Transferability, gaps, and transition scoring
# ---------------------------------------------------------------------------

def skill_gap(profile: dict[str, Any], role: dict[str, Any]) -> dict[str, Any]:
    """
    Compare profile vs a target role. Returns skills the user already has vs
    those they need to build (target requires notably more than user has).
    """
    labels = dl.skill_labels()
    have, need = [], []
    for sid, required in role.get("vector", {}).items():
        current = float(profile.get("skills", {}).get(sid, 0.0))
        gap = required - current
        entry = {
            "id": sid,
            "label": labels.get(sid, sid),
            "required": required,
            "current": current,
            "gap": round(gap, 1),
        }
        if gap >= 20 and required >= 45:  # meaningfully short on a skill that matters
            need.append(entry)
        elif current >= max(40, required - 15):
            have.append(entry)
    need.sort(key=lambda e: e["gap"], reverse=True)
    have.sort(key=lambda e: e["current"], reverse=True)
    return {"have": have, "need": need}


def transferability(profile: dict[str, Any], role: dict[str, Any]) -> dict[str, Any]:
    """
    % of the target role's required skills already met by the profile,
    with extra weight on transferable (business/analytics) skills that carry
    across roles.
    """
    transfer_ids = dl.transferable_skill_ids()
    num, den = 0.0, 0.0
    for sid, required in role.get("vector", {}).items():
        if required <= 0:
            continue
        current = float(profile.get("skills", {}).get(sid, 0.0))
        coverage = min(current / required, 1.0)
        weight = 1.5 if sid in transfer_ids else 1.0
        num += coverage * required * weight
        den += required * weight
    pct = round((num / den) * 100) if den else 0
    return {"transferable_pct": pct}


def experience_bonus(years: float) -> float:
    """Small bonus for domain experience, capped. 0 yrs -> 0, 10+ yrs -> ~1.0."""
    return min(years / 10.0, 1.0)


def transition_score(profile: dict[str, Any], role: dict[str, Any]) -> dict[str, Any]:
    """
    How realistic is moving from the profile to this role?
    Blend skill fit, skill distance (closeness), and experience.
    """
    p_vec = profile_vector(profile)
    r_vec = role_vector(role)
    distance = career_distance(p_vec, r_vec)
    shortfall = directional_shortfall(p_vec, r_vec)
    transfer = transferability(profile, role)["transferable_pct"]
    exp = experience_bonus(profile.get("years_experience", 0))

    # readiness: how close the user already is to meeting the target's bar.
    # shortfall is the "points to close"; readiness is its complement.
    readiness = max(0.0, 100.0 - shortfall)

    # weighted blend of transferable coverage, readiness, and experience
    score = (
        0.45 * transfer +
        0.40 * readiness +
        0.15 * (exp * 100)
    )
    score = round(min(max(score, 0), 100))

    return {
        "score": score,
        "skill_distance": distance,
        "skill_shortfall": shortfall,
        "readiness": round(readiness, 1),
        "transferable_pct": transfer,
        "experience_factor": round(exp, 2),
        "explanation": {
            "formula": "0.45*transferable% + 0.40*(100 - skill_shortfall) + 0.15*(experience_factor*100)",
            "transferable_pct": transfer,
            "skill_shortfall": shortfall,
            "readiness": round(readiness, 1),
            "skill_distance": distance,
            "experience_factor": round(exp, 2),
        },
    }


# ---------------------------------------------------------------------------
# 4. Future Fit + Remote Fit
# ---------------------------------------------------------------------------

FUTURE_FIT_WEIGHTS = {
    "skill_fit": 0.30,      # transition score (transferability + closeness)
    "demand": 0.20,         # current market demand
    "growth": 0.20,         # projected growth
    "ai_resilience": 0.20,  # augmented rather than automated
    "salary": 0.10,         # relative salary potential
}


def future_fit(profile: dict[str, Any], role: dict[str, Any],
               transition: dict[str, Any]) -> dict[str, Any]:
    """
    Composite 'is this a good destination?' score across five dimensions.
    Remote fit is reported separately so the user can weigh it by preference.
    """
    components = {
        "skill_fit": transition["score"],
        "demand": role.get("demand", 0),
        "growth": role.get("growth", 0),
        "ai_resilience": role.get("ai_resilience", 0),
        "salary": role.get("salary_index", 0),
    }
    overall = round(sum(components[k] * w for k, w in FUTURE_FIT_WEIGHTS.items()))
    return {
        "score": overall,
        "components": components,
        "weights": FUTURE_FIT_WEIGHTS,
        "explanation": {
            "formula": " + ".join(f"{w}*{k}" for k, w in FUTURE_FIT_WEIGHTS.items()),
            "components": components,
        },
    }


def remote_fit(profile: dict[str, Any], role: dict[str, Any]) -> dict[str, Any]:
    """Remote suitability of the target role vs the user's preference."""
    share = role.get("remote_share", 0)
    prefers_remote = bool(profile.get("remote_preference", False))
    note = (
        "Strong remote availability." if share >= 70 else
        "Moderate remote availability." if share >= 55 else
        "Limited remote availability."
    )
    if prefers_remote and share < 55:
        note += " May conflict with your remote preference."
    return {"score": share, "prefers_remote": prefers_remote, "note": note}


# ---------------------------------------------------------------------------
# 5. Roadmap + portfolio projects
# ---------------------------------------------------------------------------

def _skill_learning_action(sid: str) -> str:
    actions = {
        "python": "Python for data work (pandas, typing, testing)",
        "git_cicd": "Git workflows + basic CI/CD",
        "dbt_semantic": "dbt models + a semantic/metrics layer",
        "data_engineering": "Batch & streaming ETL fundamentals",
        "cloud": "One cloud platform (storage, compute, IAM basics)",
        "genai_llm": "LLM fundamentals and the major model APIs",
        "rag": "Retrieval-augmented generation + vector search",
        "ai_agents": "Agentic patterns (tools, planning, orchestration)",
        "prompt_eng": "Structured prompting and prompt evaluation",
        "machine_learning": "Core ML modelling and validation",
        "mlops": "Model deployment, monitoring, and MLOps",
        "software_eng": "Software engineering practices (design, review)",
        "product_mgmt": "Product discovery and lifecycle for AI products",
        "data_modeling": "Dimensional + analytical data modeling",
    }
    return actions.get(sid, dl.skill_labels().get(sid, sid))


def build_roadmap(gap: dict[str, Any]) -> dict[str, Any]:
    """Split the top skill gaps into a 3 x 30-day plan."""
    needed = [g["id"] for g in gap["need"]]
    # order by category so foundational/engineering skills come first
    cats = dl.skill_categories()
    order = {"engineering": 0, "data": 1, "ai": 2, "analytics": 3, "business": 4}
    needed.sort(key=lambda s: order.get(cats.get(s, ""), 5))

    thirds = [needed[i::3] for i in range(3)] if needed else [[], [], []]
    # distribute more sensibly: first phase = foundations
    n = len(needed)
    p1 = needed[: max(1, n // 3)] if n else []
    p2 = needed[len(p1): len(p1) + max(1, n // 3)] if n else []
    p3 = needed[len(p1) + len(p2):] if n else []

    def phase(ids: list[str]) -> list[str]:
        return [_skill_learning_action(s) for s in ids]

    return {
        "days_1_30": {"theme": "Foundations", "focus": phase(p1) or ["Reinforce existing strengths"]},
        "days_31_60": {"theme": "Core new skills", "focus": phase(p2) or ["Deepen applied practice"]},
        "days_61_90": {"theme": "Build & evaluate", "focus": phase(p3) or ["Ship a capstone project"]},
    }


PORTFOLIO_PROJECTS = {
    "analytics_ai": "Build an AI analytics agent that turns business questions into validated SQL and an executive insight summary.",
    "engineering": "Build an end-to-end dbt + orchestration pipeline with tests, docs, and CI.",
    "product": "Write an AI product spec + build a working prototype with an evaluation harness.",
    "management": "Design an analytics operating model and a metrics governance framework with a demo.",
    "science": "Build a forecasting/ML project with a clear validation and monitoring story.",
    "architecture": "Design and prototype a RAG reference architecture with evaluation and guardrails.",
    "analytics": "Build a self-serve BI + semantic layer with an AI question-answering front end.",
}


def portfolio_project(role: dict[str, Any]) -> str:
    return PORTFOLIO_PROJECTS.get(
        role.get("family", ""),
        "Build a hands-on project that demonstrates the target role's core skills end to end.",
    )


def estimate_transition_time(transition_score_val: int) -> str:
    if transition_score_val >= 85:
        return "3-6 months"
    if transition_score_val >= 75:
        return "4-8 months"
    if transition_score_val >= 60:
        return "6-12 months"
    return "9-18 months"


# ---------------------------------------------------------------------------
# 6. Top-level orchestration
# ---------------------------------------------------------------------------

# Which role families belong to each growth direction the user can pick.
DIRECTION_FAMILIES = {
    "ic_tech":    {"engineering", "architecture", "science", "analytics_ai"},
    "ic_nontech": {"analytics", "product", "risk", "finance"},
    "people":     {"management"},
}


def _direction_boost(role: dict[str, Any], direction: str | None) -> float:
    """
    Ranking nudge toward the user's chosen track. Roles whose family is on the
    chosen axis get a positive boost; clearly off-axis roles get a small penalty.
    Returns a delta added to the rank key (kept modest so fit still dominates).
    """
    if not direction or direction not in DIRECTION_FAMILIES:
        return 0.0
    fam = role.get("family")
    if fam in DIRECTION_FAMILIES[direction]:
        return 12.0
    # people <-> IC are the strongest opposites; penalize the cross a bit more
    if direction == "people" and fam in (DIRECTION_FAMILIES["ic_tech"] | DIRECTION_FAMILIES["ic_nontech"]):
        return -6.0
    if direction in ("ic_tech", "ic_nontech") and fam == "management":
        return -6.0
    return 0.0


def recommend_transitions(profile: dict[str, Any], top_n: int = 5,
                          direction: str | None = None) -> list[dict[str, Any]]:
    """
    Score every seed role as a destination and rank them. If `direction` is set
    (ic_tech / ic_nontech / people), recommendations are nudged toward that track.
    """
    current = closest_role(profile)
    current_id = current["role"]["id"]

    # Restrict candidates to roles that are ONE STEP AWAY in the business
    # task-flow (adjacency graph), so we only suggest realistic, workflow-close
    # moves - not roles that are merely skill-similar but far in the flow.
    adjacency = dl.load_role_adjacency()["neighbors"]
    neighbor_ids = set(adjacency.get(current_id, []))

    results = []
    for role in dl.load_roles()["roles"]:
        if role["id"] == current_id:
            continue  # don't recommend the role they're already in
        # if we have an adjacency list for the current role, only consider its
        # task-flow neighbors; otherwise fall back to all roles.
        if neighbor_ids and role["id"] not in neighbor_ids:
            continue
        trans = transition_score(profile, role)
        fit = future_fit(profile, role, trans)
        gap = skill_gap(profile, role)
        rfit = remote_fit(profile, role)
        results.append({
            "role_id": role["id"],
            "title": role["title"],
            "family": role.get("family"),
            "onet_code": role.get("onet_code"),
            "transition_score": trans["score"],
            "future_fit": fit["score"],
            "remote_fit": rfit["score"],
            "skill_distance": trans["skill_distance"],
            "transferable_pct": trans["transferable_pct"],
            "estimated_time": estimate_transition_time(trans["score"]),
            "skill_gap": gap,
            "roadmap": build_roadmap(gap),
            "portfolio_project": portfolio_project(role),
            "remote_note": rfit["note"],
            "future_fit_detail": fit,
            "transition_detail": trans,
            "market": {
                "demand": role.get("demand"),
                "growth": role.get("growth"),
                "remote_share": role.get("remote_share"),
                "ai_resilience": role.get("ai_resilience"),
                "salary_index": role.get("salary_index"),
            },
        })

    # rank primarily by future fit, blend with transition realism, then nudge
    # toward the user's chosen growth direction.
    results.sort(
        key=lambda r: (0.6 * r["future_fit"] + 0.4 * r["transition_score"]
                       + _direction_boost({"family": r["family"]}, direction)),
        reverse=True,
    )
    return results[:top_n]


def analyze_profile(profile: dict[str, Any], top_n: int = 5,
                    direction: str | None = None) -> dict[str, Any]:
    """Full Career Navigator analysis for a user profile."""
    return {
        "profile": {
            "title": profile.get("title"),
            "years_experience": profile.get("years_experience"),
            "industry": profile.get("industry"),
            "location": profile.get("location"),
            "remote_preference": profile.get("remote_preference"),
        },
        "skill_profile": classify_skills(profile),
        "current_role_match": closest_role(profile)["role"]["title"],
        "ai_transformation": role_transformation_score(profile),
        "recommendations": recommend_transitions(profile, top_n=top_n, direction=direction),
    }
