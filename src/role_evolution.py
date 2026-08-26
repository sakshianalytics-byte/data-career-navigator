"""
role_evolution.py
-----------------
Workflow-grounded "how your role is changing" analysis, built on top of the
deterministic engine. Two capabilities:

1. seniority_shift(role)
   Reads a role's production_exposure vs judgment_exposure and derives what the
   AI shift means for hiring / seniority:
     - production-heavy + low judgment  -> the role DE-LEVELS (juniors + AI cover
       the production work); an experienced person's edge is to move up into the
       judgment/merged layer.
     - judgment-heavy                    -> the role CONSOLIDATES upward into
       senior merged roles; experience is the moat.

2. merge_options(profile, source_role, chosen_axis=None)
   For the user's matched role, returns the Tech / Business / People merge forks
   from the curated map (data/role_merges.json). Each fork is a SYNTHESIZED
   merged role: the residual-human skills of the source + partner (the parts AI
   does NOT absorb), unioned with AI-orchestration "glue" skills. Returns a skill
   gap vs the user's current profile and a rationale. If an axis is chosen, that
   fork is flagged as recommended.

All illustrative (seed data); the METHOD is the point. No network, no LLM.
"""

from __future__ import annotations

from typing import Any

from . import data_loader as dl
from . import engine as E


# ---------------------------------------------------------------------------
# 1. Seniority-shift signal
# ---------------------------------------------------------------------------

def seniority_shift(role: dict[str, Any]) -> dict[str, Any]:
    """Derive the de-level vs consolidate signal from the exposure split."""
    prod = role.get("production_exposure", 0)
    judg = role.get("judgment_exposure", 0)

    # A role de-levels when a lot of its work is automatable production and the
    # protected judgment core is small. It consolidates upward when judgment is
    # the dominant remaining human contribution.
    if prod >= 55 and judg <= 30:
        pattern = "de-leveling"
        headline = "This role is de-leveling."
        detail = (
            f"About {prod}% of the work is automatable production, and only ~{judg}% is "
            "protected judgement. Expect more of this role to be done by less-experienced "
            "people working with AI. If you are experienced, your highest-ROI move is up "
            "into the judgement/merged layer - don't compete with juniors-plus-AI on production."
        )
    elif judg >= 25:
        pattern = "consolidating"
        headline = "This role is consolidating upward."
        detail = (
            f"Its judgement core (~{judg}%) is hard for AI to own, so the role tends to "
            "consolidate into senior, merged seats that supervise AI output and own the "
            "decisions. Your experience is the moat - grow into the accountable, cross-functional version."
        )
    else:
        pattern = "augmenting"
        headline = "This role is being augmented, not replaced."
        detail = (
            f"With ~{prod}% production exposure and ~{judg}% judgement, AI mostly speeds you up. "
            "Add AI-orchestration skills to stay ahead and move toward the higher-judgement version."
        )

    return {
        "pattern": pattern,
        "headline": headline,
        "detail": detail,
        "production_exposure": prod,
        "judgment_exposure": judg,
    }


# ---------------------------------------------------------------------------
# 2. Residual-human skills + merge synthesis
# ---------------------------------------------------------------------------

def residual_human_vector(role: dict[str, Any]) -> dict[str, float]:
    """
    The part of a role's skill demand that stays human after AI absorbs the
    production layer. We scale each skill by the role's judgement share: a role
    with a bigger protected judgement core keeps more of its skill requirements
    as human-owned, while production-heavy roles keep proportionally less.

    kept_fraction = 1 - production_exposure/100, floored so nothing vanishes.
    """
    prod = role.get("production_exposure", 0)
    kept = max(0.35, 1.0 - prod / 100.0)  # keep at least 35% so skills don't zero out
    return {sid: round(val * kept, 1) for sid, val in role.get("vector", {}).items()}


def _blend_vectors(*vectors: dict[str, float]) -> dict[str, float]:
    """Element-wise max across skill maps (union that keeps the strongest demand)."""
    out: dict[str, float] = {}
    for vec in vectors:
        for sid, val in vec.items():
            out[sid] = max(out.get(sid, 0.0), float(val))
    return out


def synthesize_merged_role(source: dict[str, Any], partner: dict[str, Any],
                           title: str) -> dict[str, Any]:
    """
    Build a merged role from the residual-human skills of source + partner,
    unioned with the AI-orchestration glue skills. The merged role inherits the
    stronger market signals of the two parents (you're moving up, not sideways).
    """
    merges = dl.load_role_merges()
    glue_ids = merges["glue_skills"]

    src_res = residual_human_vector(source)
    par_res = residual_human_vector(partner)
    merged_vec = _blend_vectors(src_res, par_res)

    # Ensure the AI-orchestration glue skills are meaningfully required - these
    # are what let one person supervise AI across the merged workflow.
    for gid in glue_ids:
        merged_vec[gid] = max(merged_vec.get(gid, 0.0), 65.0)

    def better(key: str) -> Any:
        return max(source.get(key, 0), partner.get(key, 0))

    return {
        "id": f"merged::{source['id']}+{partner['id']}",
        "title": title,
        "family": "merged",
        "onet_code": partner.get("onet_code") or source.get("onet_code"),
        "seniority": "senior",
        "vector": {k: round(v) for k, v in merged_vec.items()},
        "demand": better("demand"),
        "growth": better("growth"),
        "remote_share": better("remote_share"),
        "ai_resilience": better("ai_resilience"),
        "salary_index": better("salary_index"),
        "glue_skills": glue_ids,
    }


def _fallback_partner(source_id: str, axis: str) -> str | None:
    """
    If a role isn't in the curated merge map, pick a reasonable partner by axis
    using role families (keeps the feature working for any matched role).
    """
    roles = dl.roles_by_id()
    prefer = {
        "tech": ["ai_engineer", "analytics_engineer", "ml_engineer"],
        "business": ["product_manager", "ai_product_manager", "strategy_manager"],
        "people": ["analytics_manager"],
    }[axis]
    for rid in prefer:
        if rid in roles and rid != source_id:
            return rid
    return None


def merge_options(profile: dict[str, Any], source_role: dict[str, Any],
                  chosen_axis: str | None = None) -> dict[str, Any]:
    """
    Return the three direction forks for the user's matched role. Each fork
    carries the synthesized merged role, a skill gap vs the profile, and the
    curated rationale. `chosen_axis` (if given) flags the recommended fork.
    """
    merges = dl.load_role_merges()
    directions = merges["directions"]
    src_id = source_role["id"]
    src_map = merges["merges"].get(src_id)
    roles = dl.roles_by_id()

    forks = []
    for axis, meta in directions.items():
        spec = src_map.get(axis) if src_map else None
        if spec and spec["partner"] in roles:
            partner = roles[spec["partner"]]
            title = spec["title"]
            rationale = spec["rationale"]
        else:
            pid = _fallback_partner(src_id, axis)
            if not pid:
                continue
            partner = roles[pid]
            title = f"{source_role['title']} + {partner['title']} (merged)"
            rationale = (f"AI absorbs the production overlap of {source_role['title']} and "
                         f"{partner['title']}, so one person can own the combined workflow.")

        merged = synthesize_merged_role(source_role, partner, title)
        trans = E.transition_score(profile, merged)
        gap = E.skill_gap(profile, merged)
        forks.append({
            "axis": axis,
            "axis_label": meta["label"],
            "axis_meaning": meta["meaning"],
            "partner_role": partner["title"],
            "merged_title": title,
            "rationale": rationale,
            "transition_score": trans["score"],
            "estimated_time": E.estimate_transition_time(trans["score"]),
            "skill_gap_need": [g["label"] for g in gap["need"][:6]],
            "merged_role": merged,
            "recommended": (chosen_axis is not None and axis == chosen_axis),
        })

    return {
        "source_role": source_role["title"],
        "chosen_axis": chosen_axis,
        "forks": forks,
    }


def analyze_role_evolution(profile: dict[str, Any],
                           chosen_axis: str | None = None) -> dict[str, Any]:
    """Top-level: matched role -> seniority signal + the three merge forks."""
    matched = E.closest_role(profile)["role"]
    return {
        "matched_role": matched["title"],
        "seniority_shift": seniority_shift(matched),
        "merge": merge_options(profile, matched, chosen_axis=chosen_axis),
    }
