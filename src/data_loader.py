"""
data_loader.py
--------------
Loads the bundled open seed dataset (roles, skills, AI task impact).

To make the engine authoritative, replace the JSON files in ../data with
data derived from O*NET (tasks/skills), BLS (growth/demand) and an
open-licensed job-postings dataset (market demand / remote share), keeping
the same schema described below.

Schemas
-------
skills.json   -> { "skills": [{id, label, category}], transferable_categories, technical_categories }
roles.json    -> { "roles": [{id, title, family, onet_code, seniority, vector{skill_id:0-100},
                              demand, growth, remote_share, ai_resilience, salary_index}] }
ai_task_impact.json -> { impact_levels{...}, "tasks": [{id, label, impact, exposure, keywords[], human_value}] }
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

# data/ lives next to src/ at the project root
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_json(name: str) -> dict[str, Any]:
    path = DATA_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_skills() -> dict[str, Any]:
    """Canonical skill dimensions + category groupings."""
    return _read_json("skills.json")


@lru_cache(maxsize=1)
def load_roles() -> dict[str, Any]:
    """
    Role skill vectors + market signals.

    Backfills production_exposure / judgment_exposure for any role missing them
    so the whole dataset is consistent. Illustrative defaults are derived from
    ai_resilience: a more AI-resilient role has less of its work automatable
    (lower production exposure) and a larger protected judgment core.
    """
    data = _read_json("roles.json")
    for role in data["roles"]:
        resilience = role.get("ai_resilience", 50)
        if "production_exposure" not in role:
            # less resilient -> more of the production work is exposed to AI
            role["production_exposure"] = max(0, min(100, 100 - resilience + 15))
        if "judgment_exposure" not in role:
            # the judgment core is small and shrinks as resilience rises
            role["judgment_exposure"] = max(0, min(100, round((100 - resilience) * 0.4)))
    return data


@lru_cache(maxsize=1)
def load_task_impact() -> dict[str, Any]:
    """AI task archetypes for the transformation table and JD analyzer."""
    return _read_json("ai_task_impact.json")


@lru_cache(maxsize=1)
def load_role_merges() -> dict[str, Any]:
    """Curated Tech/Business/People merge map for the role-evolution feature."""
    return _read_json("role_merges.json")


def skill_ids() -> list[str]:
    """Ordered list of canonical skill ids (defines vector dimension order)."""
    return [s["id"] for s in load_skills()["skills"]]


def skill_labels() -> dict[str, str]:
    return {s["id"]: s["label"] for s in load_skills()["skills"]}


def skill_categories() -> dict[str, str]:
    return {s["id"]: s["category"] for s in load_skills()["skills"]}


def transferable_skill_ids() -> set[str]:
    cats = set(load_skills()["transferable_categories"])
    return {s["id"] for s in load_skills()["skills"] if s["category"] in cats}


def roles_by_id() -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in load_roles()["roles"]}


def get_role(role_id: str) -> dict[str, Any]:
    roles = roles_by_id()
    if role_id not in roles:
        raise KeyError(f"Unknown role id '{role_id}'. Known: {sorted(roles)}")
    return roles[role_id]
