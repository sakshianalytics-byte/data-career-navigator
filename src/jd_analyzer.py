"""
jd_analyzer.py
--------------
Job Description AI-Exposure Analyzer.

Two entry points:

1. analyze_jd(text) -> classify a free-text job description into:
     - overall "AI takeover %"
     - actions AI can take over
     - actions that remain human-critical
     - human + AI hybrid tasks
     - a task-level breakdown table with rationale

2. score_tasks_for_role(role) -> used by engine.role_transformation_score to
   build the current-role AI transformation table from a role's skill vector.

Deterministic and offline: JD parsing uses sentence splitting + keyword
matching against the task archetypes in data/ai_task_impact.json. An optional
LLM layer can be plugged in later for richer parsing, but is never required.
"""

from __future__ import annotations

import re
from typing import Any

from . import data_loader as dl

# ---------------------------------------------------------------------------
# Task archetype access
# ---------------------------------------------------------------------------

def _tasks() -> list[dict[str, Any]]:
    return dl.load_task_impact()["tasks"]


def _impact_levels() -> dict[str, Any]:
    return dl.load_task_impact()["impact_levels"]


# skill_id -> which task archetypes that skill drives (for role-based table)
_SKILL_TO_TASKS: dict[str, list[str]] = {
    "sql": ["sql_generation", "data_cleaning"],
    "data_modeling": ["coding_implementation", "data_pipeline_ops"],
    "bi_reporting": ["recurring_reporting", "dashboard_creation"],
    "data_viz": ["dashboard_creation"],
    "statistics": ["exploratory_analysis", "model_building"],
    "experimentation": ["exploratory_analysis", "model_building"],
    "python": ["coding_implementation", "data_cleaning"],
    "software_eng": ["coding_implementation", "data_pipeline_ops"],
    "git_cicd": ["data_pipeline_ops"],
    "cloud": ["data_pipeline_ops"],
    "data_engineering": ["data_pipeline_ops", "data_cleaning"],
    "dbt_semantic": ["coding_implementation", "data_pipeline_ops"],
    "machine_learning": ["model_building"],
    "mlops": ["data_pipeline_ops", "model_building"],
    "genai_llm": ["ai_evaluation_task", "coding_implementation"],
    "rag": ["ai_evaluation_task"],
    "ai_agents": ["ai_evaluation_task"],
    "prompt_eng": ["ai_evaluation_task"],
    "business_analysis": ["business_interpretation", "problem_framing"],
    "stakeholder_mgmt": ["stakeholder_management"],
    "problem_framing": ["problem_framing"],
    "product_mgmt": ["problem_framing", "stakeholder_management"],
    "domain_knowledge": ["business_interpretation"],
    "people_mgmt": ["leadership_mentoring"],
    "operations": ["problem_framing", "leadership_mentoring"],
    "strategy": ["problem_framing"],
    "documentation": ["documentation"],
    "data_governance": ["governance_compliance"],
    "risk_analysis": ["exploratory_analysis", "governance_compliance"],
    "regulatory_compliance": ["governance_compliance"],
    "financial_analysis": ["exploratory_analysis", "business_interpretation"],
    "business_requirements": ["problem_framing", "documentation"],
}


# ---------------------------------------------------------------------------
# Role-based transformation table (used by engine)
# ---------------------------------------------------------------------------

# Technical skills whose contribution to the task table is down-weighted for
# management-family roles: a manager who "knows SQL" mostly oversees/reviews
# hands-on work rather than producing it, so it should not dominate their
# AI-transformation task mix.
_TECHNICAL_SKILLS = {
    "sql", "python", "software_eng", "git_cicd", "cloud", "data_engineering",
    "dbt_semantic", "data_modeling", "mlops", "machine_learning", "bi_reporting",
    "data_viz", "rag", "ai_agents",
}
# Families whose hands-on technical skills are down-weighted in the task table
# (managers and product roles mostly oversee/decide rather than produce).
_MANAGEMENT_FAMILIES = {"management", "product"}
# multiplier applied to technical skills' task contribution for those families
_MGMT_TECH_WEIGHT = 0.35


def score_tasks_for_role(role: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build a task-level AI-impact table for a role from its skill vector.

    A task's weight = how strongly the role's skills drive that task. We only
    include tasks whose driving skills are meaningfully present in the role.

    For management-family roles, technical skills contribute at a reduced weight
    so their task mix reflects leadership/oversight work rather than hands-on IC
    production they merely supervise.
    """
    tasks_by_id = {t["id"]: t for t in _tasks()}
    levels = _impact_levels()
    vector = role.get("vector", {})
    is_mgmt = role.get("family") in _MANAGEMENT_FAMILIES

    weights: dict[str, float] = {}
    always_keep: set[str] = set()  # tasks driven by a standout (>=70) skill
    for sid, val in vector.items():
        contribution = float(val)
        if is_mgmt and sid in _TECHNICAL_SKILLS:
            contribution *= _MGMT_TECH_WEIGHT
        for task_id in _SKILL_TO_TASKS.get(sid, []):
            weights[task_id] = weights.get(task_id, 0.0) + contribution
            # A NON-technical skill the role is strong in should always surface
            # its task (e.g. People Management for a manager), even if a single
            # skill drives it, so it isn't dropped by the relative-weight filter.
            # We exclude technical skills so a manager's residual reporting/SQL
            # doesn't force IC production tasks back into their view.
            if val >= 70 and sid not in _TECHNICAL_SKILLS:
                always_keep.add(task_id)

    # Drop tasks whose weight is trivial relative to the role's strongest task,
    # so a role only shows work it meaningfully does (e.g. a manager stops
    # showing low-weight IC technical tasks they merely oversee).
    max_weight = max(weights.values(), default=0.0)
    min_keep = max(40.0, 0.45 * max_weight)

    rows = []
    for task_id, weight in weights.items():
        if weight < min_keep and task_id not in always_keep:
            continue  # too weak, and not driven by a standout skill
        task = tasks_by_id[task_id]
        level = task["impact"]
        rows.append({
            "task_id": task_id,
            "task": task["label"],
            "impact": level,
            "impact_label": levels[level]["label"],
            "color": levels[level]["color"],
            "exposure": task["exposure"],
            "weight": round(weight, 1),
            "human_value": task["human_value"],
            "what_happens": levels[level]["meaning"],
        })
    rows.sort(key=lambda r: r["exposure"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Free-text JD parsing
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split a JD into candidate task lines: by newline, bullet, and sentence."""
    # normalize bullets and semicolons into line breaks
    text = re.sub(r"[•\u2022\-\*]\s+", "\n", text)
    parts = re.split(r"[\n\r]+|(?<=[.;])\s+", text)
    lines = [p.strip() for p in parts if len(p.strip()) >= 12]
    return lines


def _match_tasks_in_line(line: str) -> list[dict[str, Any]]:
    """Return every task archetype whose keywords appear in the line."""
    low = line.lower()
    matches = []
    for task in _tasks():
        for kw in task["keywords"]:
            if kw in low:
                matches.append(task)
                break
    return matches


def analyze_jd(text: str) -> dict[str, Any]:
    """
    Analyze a free-text job description.

    Returns overall AI takeover %, categorized action lists, and a task-level
    breakdown with rationale. Every number is explainable.
    """
    levels = _impact_levels()
    lines = _split_sentences(text)

    # dedupe matched tasks but keep an example line for each
    matched: dict[str, dict[str, Any]] = {}
    for line in lines:
        for task in _match_tasks_in_line(line):
            if task["id"] not in matched:
                matched[task["id"]] = {"task": task, "example": line}

    breakdown = []
    for entry in matched.values():
        task = entry["task"]
        level = task["impact"]
        breakdown.append({
            "task_id": task["id"],
            "task": task["label"],
            "impact": level,
            "impact_label": levels[level]["label"],
            "color": levels[level]["color"],
            "exposure": task["exposure"],
            "what_happens": levels[level]["meaning"],
            "human_value": task["human_value"],
            "matched_text": entry["example"],
        })
    breakdown.sort(key=lambda r: r["exposure"], reverse=True)

    if breakdown:
        takeover = round(sum(r["exposure"] for r in breakdown) / len(breakdown))
    else:
        takeover = 0

    ai_takeover_actions = [
        r for r in breakdown if r["impact"] in ("automated", "ai_assisted")
    ]
    hybrid_actions = [r for r in breakdown if r["impact"] == "ai_augmented"]
    human_actions = [r for r in breakdown if r["impact"] == "human_critical"]

    return {
        "ai_takeover_pct": takeover,
        "summary": _summary(takeover, len(breakdown)),
        "tasks_detected": len(breakdown),
        "ai_takeover_actions": ai_takeover_actions,
        "hybrid_actions": hybrid_actions,
        "human_critical_actions": human_actions,
        "breakdown": breakdown,
        "explanation": {
            "method": (
                "The JD is split into task lines and matched against known task "
                "archetypes by keyword. The AI takeover % is the average AI "
                "exposure across the detected tasks. Framed as task transformation, "
                "not job elimination."
            ),
            "formula": "mean(exposure of detected tasks)",
        },
    }


def _summary(takeover: int, n_tasks: int) -> str:
    if n_tasks == 0:
        return ("No recognizable tasks were detected. Try pasting the "
                "responsibilities / requirements section of the JD.")
    if takeover >= 70:
        band = "highly exposed to AI transformation"
    elif takeover >= 50:
        band = "substantially reshaped by AI, with humans reviewing and directing"
    elif takeover >= 35:
        band = "AI-augmented, with humans firmly in control"
    else:
        band = "anchored in human judgement, with AI in a supporting role"
    return (f"About {takeover}% of this role's detected tasks are exposed to AI. "
            f"The role is {band}.")
