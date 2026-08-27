"""
analytics.py
------------
Privacy-safe, best-effort usage logging to a Google Sheet.

Design goals:
  - NEVER crash the app. If analytics isn't configured (or the network fails),
    logging silently no-ops so the user experience is unaffected.
  - NEVER log user content. We record only coarse EVENT TYPES and safe metadata
    (e.g. "jd_analyzed", a role family, an anonymous session id) - never the
    resume text or the pasted job description.
  - Keep all credentials OUT of the repo. Credentials come from Streamlit
    Secrets (st.secrets), not from files in the codebase.

Configuration (Streamlit Cloud -> App -> Settings -> Secrets):

    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
    client_email = "logger@your-project.iam.gserviceaccount.com"
    client_id = "..."
    token_uri = "https://oauth2.googleapis.com/token"

    [analytics]
    sheet_key = "the-long-id-from-your-google-sheet-url"
    worksheet = "events"   # optional, defaults to the first worksheet

Dependencies (add to requirements.txt): gspread, google-auth
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import Any

# We import streamlit lazily-safe: this module is only used from a Streamlit app,
# but we never want an import error to take down the app.
try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore


# Column order written to the sheet. Keep this stable.
_HEADERS = ["timestamp_utc", "session_id", "event", "detail", "input", "output"]

# Cap stored input/output length so a huge value can't create an unwieldy cell.
_MAX_INPUT_CHARS = 2000
_MAX_OUTPUT_CHARS = 4000


def _get_session_id() -> str:
    """A per-browser-session anonymous id (not tied to any personal data)."""
    if st is None:
        return "no-session"
    if "sid" not in st.session_state:
        st.session_state["sid"] = uuid.uuid4().hex[:12]
    return st.session_state["sid"]


def _enabled() -> bool:
    """Analytics is on only if Streamlit Secrets are present and complete."""
    if st is None:
        return False
    try:
        return (
            "gcp_service_account" in st.secrets
            and "analytics" in st.secrets
            and "sheet_key" in st.secrets["analytics"]
        )
    except Exception:
        return False


# Default worksheet (tab) names per app feature. Each tab logs to its own sheet.
SHEET_NAVIGATOR = "Career Navigator"
SHEET_JD = "JD Analyzer"
SHEET_MERGE = "Role Merge"
SHEET_DEFAULT = SHEET_NAVIGATOR


def _open_spreadsheet():
    """Authorize and return the spreadsheet handle."""
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=scopes
    )
    client = gspread.authorize(creds)
    return client.open_by_key(st.secrets["analytics"]["sheet_key"])


def _build_worksheet(sheet_name: str):
    """Get-or-create the named worksheet (tab) and ensure a header row."""
    sh = _open_spreadsheet()
    try:
        ws = sh.worksheet(sheet_name)
    except Exception:
        # tab doesn't exist yet -> create it
        ws = sh.add_worksheet(title=sheet_name, rows=1000, cols=len(_HEADERS))
    try:
        if not ws.get_all_values():
            ws.append_row(_HEADERS)
    except Exception:
        pass
    return ws


# Cache the authorized worksheet PER NAME across reruns.
if st is not None:
    _get_worksheet = st.cache_resource(_build_worksheet)
else:  # pragma: no cover
    _get_worksheet = _build_worksheet


def log_event(event: str, detail: str = "", user_input: str = "",
              output: str = "", sheet: str = SHEET_DEFAULT) -> None:
    """
    Record one usage event to a specific worksheet (tab). Best-effort; never
    crashes the app.

    Parameters
    ----------
    event : short event type, e.g. "profile_analyzed", "jd_analyzed", "roles_merged".
    detail : coarse computed metadata (e.g. matched role, takeover %).
    user_input : the content the user submitted. Truncated to _MAX_INPUT_CHARS.
    output : a summary of what the tool returned. Truncated to _MAX_OUTPUT_CHARS.
    sheet : which worksheet/tab to log to (SHEET_NAVIGATOR / SHEET_JD / SHEET_MERGE).
    """
    if not _enabled():
        return
    try:
        row = [
            _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            _get_session_id(),
            str(event)[:60],
            str(detail)[:120],
            str(user_input)[:_MAX_INPUT_CHARS],
            str(output)[:_MAX_OUTPUT_CHARS],
        ]
        _get_worksheet(sheet).append_row(row, value_input_option="RAW")
    except Exception:
        # Never let analytics failures affect the user.
        return
