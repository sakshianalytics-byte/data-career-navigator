"""
role_evolution.py
-----------------
The "your role can merge into X and become Y" feature, built on the deterministic
engine. Given a matched role and a growth direction (ic_tech / ic_nontech / people),
it synthesizes a MERGED future role and lists what to learn to get there.

Capabilities
------------
1. seniority_shift(role)
   From production_exposure vs judgment_exposure, describes whether the role
   de-levels (juniors + AI), consolidates upward, or is merely augmented.

2. merged_role_for(profile, source_role, direction)
   Returns ONE synthesized merged role for the chosen direction: the residual
   human skills of source + partner (the parts AI does NOT absorb), unioned with
   AI-orchestration "glue" skills, plus a skill gap vs the profile and rationale.

3. analyze_role_evolution(profile, direction)
   Top-level: matched role + seniority signal + the merged future role.

All illustrative (seed data). No network, no LLM. Deterministic.
"""

from __future__ import annotations

from typing import Any

from . import data_loader as dl
from . import engine as E


# ---------------------------------------------------------------------------
# 1. Seniority-shift signal
# ---------------------------------------------------------------------------

def seniority_shift(role: dict[str, Any]) -> dict[str, Any]:
    prod = role.get("production_exposure", 0)
    judg = role.get("judgment_exposure", 0)
    if prod >= 55 and judg <= 30:
        pattern = "de-leveling"
        headline = "This role is de-leveling."
        detail = (f"About {prod}% of the work is automatable production and only ~{judg}% is "
                  "protected judgement, so more of it will be done by less-experienced people "
                  "plus AI. Your highest-ROI move is up into a merged, judgement-heavy role.")
    elif judg >= 25:
        pattern = "consolidating"
        headline = "This role is consolidating upward."
        detail = (f"Its judgement core (~{judg}%) is hard for AI to own, so it consolidates into "
                  "senior merged seats that supervise AI output and own decisions. Your experience is the moat.")
    else:
        pattern = "augmenting"
        headline = "This role is being augmented, not replaced."
        detail = (f"With ~{prod}% production exposure and ~{judg}% judgement, AI mostly speeds you up. "
                  "Add AI-orchestration skills and move toward the higher-judgement merged role.")
    return {"pattern": pattern, "headline": headline, "detail": detail,
            "production_exposure": prod, "judgment_exposure": judg}


# ---------------------------------------------------------------------------
# 2. Residual-human skills + merge synthesis
# ---------------------------------------------------------------------------

def residual_human_vector(role: dict[str, Any]) -> dict[str, float]:
    """Skill demand that stays human after AI absorbs the production layer."""
    prod = role.get("production_exposure", 0)
    kept = max(0.35, 1.0 - prod / 100.0)
    return {sid: round(val * kept, 1) for sid, val in role.get("vector", {}).items()}


def _blend_vectors(*vectors: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for vec in vectors:
        for sid, val in vec.items():
            out[sid] = max(out.get(sid, 0.0), float(val))
    return out


def synthesize_merged_role(source: dict[str, Any], partner: dict[str, Any],
                           title: str) -> dict[str, Any]:
    """Merged role = residual-human(source) + residual-human(partner) + glue skills."""
    glue_ids = dl.load_role_merges()["glue_skills"]
    merged_vec = _blend_vectors(residual_human_vector(source), residual_human_vector(partner))
    for gid in glue_ids:
        merged_vec[gid] = max(merged_vec.get(gid, 0.0), 65.0)

    def better(key: str) -> Any:
        return max(source.get(key, 0), partner.get(key, 0))

    return {
        "id": f"merged::{source['id']}+{partner['id']}",
        "title": title, "family": "merged",
        "onet_code": partner.get("onet_code") or source.get("onet_code"),
        "seniority": "senior",
        "vector": {k: round(v) for k, v in merged_vec.items()},
        "demand": better("demand"), "growth": better("growth"),
        "remote_share": better("remote_share"), "ai_resilience": better("ai_resilience"),
        "salary_index": better("salary_index"), "glue_skills": glue_ids,
    }


def _fallback_partner(source_id: str, direction: str) -> str | None:
    """Family-based partner if the source role isn't in the curated map."""
    roles = dl.roles_by_id()
    prefer = {
        "ic_tech": ["ai_engineer", "analytics_engineer", "ml_engineer"],
        "ic_nontech": ["product_manager", "ai_product_manager", "strategy_manager"],
        "people": ["analytics_manager"],
    }.get(direction, [])
    for rid in prefer:
        if rid in roles and rid != source_id:
            return rid
    return None


def merged_role_for(profile: dict[str, Any], source_role: dict[str, Any],
                    direction: str) -> dict[str, Any] | None:
    """Synthesize the merged future role for the chosen growth direction."""
    merges = dl.load_role_merges()
    roles = dl.roles_by_id()
    src_id = source_role["id"]
    spec = merges["merges"].get(src_id, {}).get(direction)

    if spec and spec["partner"] in roles:
        partner = roles[spec["partner"]]
        title, rationale = spec["title"], spec["rationale"]
    else:
        pid = _fallback_partner(src_id, direction)
        if not pid:
            return None
        partner = roles[pid]
        title = f"{source_role['title']} + {partner['title']} (merged)"
        rationale = (f"AI absorbs the production overlap of {source_role['title']} and "
                     f"{partner['title']}, so one person can own the combined workflow.")

    merged = synthesize_merged_role(source_role, partner, title)
    trans = E.transition_score(profile, merged)
    gap = E.skill_gap(profile, merged)
    return {
        "direction": direction,
        "direction_label": merges["directions"][direction]["label"],
        "from_role": source_role["title"],
        "partner_role": partner["title"],
        "merged_title": title,
        "rationale": rationale,
        "transition_score": trans["score"],
        "estimated_time": E.estimate_transition_time(trans["score"]),
        "skills_to_learn": [g["label"] for g in gap["need"][:6]],
        "merged_role": merged,
    }


def analyze_role_evolution(profile: dict[str, Any], direction: str) -> dict[str, Any]:
    """Matched role -> seniority signal + the merged future role for a direction."""
    matched = E.closest_role(profile)["role"]
    return {
        "matched_role": matched["title"],
        "seniority_shift": seniority_shift(matched),
        "merged": merged_role_for(profile, matched, direction),
    }


# ---------------------------------------------------------------------------
# 3. Merge ANY two roles (Role Merge Explorer tab)
# ---------------------------------------------------------------------------

def _role_ai_exposure(role: dict[str, Any]) -> int:
    """Weight-averaged AI exposure across the tasks a role performs (0-100)."""
    from .jd_analyzer import score_tasks_for_role
    rows = score_tasks_for_role(role)
    if not rows:
        return 0
    num = sum(t["exposure"] * t["weight"] for t in rows)
    den = sum(t["weight"] for t in rows)
    return round(num / den) if den else 0


def _merged_title(role_a: dict[str, Any], role_b: dict[str, Any]) -> str:
    """Prefer a curated merged title if one of the roles maps to the other."""
    merges = dl.load_role_merges()["merges"]
    for src, partner in ((role_a, role_b), (role_b, role_a)):
        for spec in merges.get(src["id"], {}).values():
            if spec.get("partner") == partner["id"]:
                return spec["title"]
    # otherwise coin a name from both
    return f"{role_a['title']} × {role_b['title']} (merged)"


def merge_two_roles(role_a_id: str, role_b_id: str) -> dict[str, Any]:
    """
    Merge two arbitrary roles into one. Returns each role's task/exposure table,
    the synthesized merged role, its overall AI exposure %, the human-critical
    skills the merged role still needs, and a suggested name.
    """
    from .jd_analyzer import score_tasks_for_role
    roles = dl.roles_by_id()
    a, b = roles[role_a_id], roles[role_b_id]
    labels = dl.skill_labels()

    title = _merged_title(a, b)
    merged = synthesize_merged_role(a, b, title)

    # skills the merged role requires most (its defining human + glue skills)
    top_skills = sorted(merged["vector"].items(), key=lambda kv: kv[1], reverse=True)
    required_skills = [
        {"label": labels.get(sid, sid), "level": val}
        for sid, val in top_skills if val >= 55
    ][:10]

    return {
        "role_a": {"title": a["title"], "ai_exposure": _role_ai_exposure(a),
                   "tasks": score_tasks_for_role(a)},
        "role_b": {"title": b["title"], "ai_exposure": _role_ai_exposure(b),
                   "tasks": score_tasks_for_role(b)},
        "merged": {
            "title": title,
            "ai_exposure": _role_ai_exposure(merged),
            "tasks": score_tasks_for_role(merged),
            "required_skills": required_skills,
            "glue_skills": [labels.get(g, g) for g in merged.get("glue_skills", [])],
            "market": {
                "demand": merged.get("demand"), "growth": merged.get("growth"),
                "remote_share": merged.get("remote_share"),
                "ai_resilience": merged.get("ai_resilience"),
                "salary_index": merged.get("salary_index"),
            },
        },
    }
