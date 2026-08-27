"""
export_data.py  (one-off)
-------------------------
Exports the stored model data to an Excel workbook with multiple sheets:
  1. Role_Skill_Matrix  - every role x every skill (0-100 required proficiency)
  2. Role_AI_Tasks      - each role's AI task table (task, impact, exposure)
  3. Role_Recommendations - for each role (as the user's current role), its
                            recommended next roles + skills to build for each

Run:  python export_data.py   ->  data_career_navigator_export.xlsx
"""

import pandas as pd

from src import data_loader as dl
from src.engine import analyze_profile
from src.jd_analyzer import score_tasks_for_role

OUT = "data_career_navigator_export.xlsx"


def role_skill_matrix() -> pd.DataFrame:
    labels = dl.skill_labels()
    sids = dl.skill_ids()
    rows = []
    for r in dl.load_roles()["roles"]:
        row = {"Role": r["title"]}
        for sid in sids:
            row[labels[sid]] = r["vector"].get(sid, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def role_ai_tasks() -> pd.DataFrame:
    rows = []
    for r in dl.load_roles()["roles"]:
        for t in score_tasks_for_role(r):
            rows.append({
                "Role": r["title"],
                "Task": t["task"],
                "AI impact": t["impact_label"],
                "AI exposure (0-100)": t["exposure"],
            })
    return pd.DataFrame(rows)


def role_recommendations() -> pd.DataFrame:
    """
    Treat each role as a 'current role' profile (using its own skill vector),
    then record the recommended next roles and the skills to build for each.
    """
    rows = []
    for r in dl.load_roles()["roles"]:
        profile = {
            "title": r["title"],
            "years_experience": 8,
            "skills": dict(r["vector"]),
            "remote_preference": True,
        }
        result = analyze_profile(profile, top_n=5, direction="ic_tech")
        for rec in result["recommendations"]:
            rows.append({
                "Current role": r["title"],
                "Recommended role": rec["title"],
                "Transition score": rec["transition_score"],
                "Future fit": rec["future_fit"],
                "Est. time": rec["estimated_time"],
                "Skills to build": ", ".join(n["label"] for n in rec["skill_gap"]["need"]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        role_skill_matrix().to_excel(xl, sheet_name="Role_Skill_Matrix", index=False)
        role_ai_tasks().to_excel(xl, sheet_name="Role_AI_Tasks", index=False)
        role_recommendations().to_excel(xl, sheet_name="Role_Recommendations", index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
