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
_HEADERS = ["timestamp_utc", "session_id", "event", "detail"]


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


def _build_worksheet():
    """Authorize and return the target worksheet."""
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
    sh = client.open_by_key(st.secrets["analytics"]["sheet_key"])
    ws_name = st.secrets["analytics"].get("worksheet")
    ws = sh.worksheet(ws_name) if ws_name else sh.sheet1

    # Ensure a header row exists (only writes if the sheet is empty).
    try:
        if not ws.get_all_values():
            ws.append_row(_HEADERS)
    except Exception:
        pass
    return ws


# Cache the authorized worksheet across reruns when Streamlit is available.
if st is not None:
    _worksheet = st.cache_resource(_build_worksheet)
else:  # pragma: no cover
    _worksheet = _build_worksheet


def log_event(event: str, detail: str = "") -> None:
    """
    Record one usage event. Best-effort and privacy-safe.

    Parameters
    ----------
    event : short event type, e.g. "app_open", "profile_analyzed", "jd_analyzed",
            "explain_requested". Use a small, fixed vocabulary.
    detail : coarse, non-personal metadata only (e.g. a role family or a bucketed
             number). Do NOT pass resume/JD text here.
    """
    if not _enabled():
        return
    try:
        row = [
            _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            _get_session_id(),
            str(event)[:60],
            str(detail)[:120],
        ]
        _worksheet().append_row(row, value_input_option="RAW")
    except Exception:
        # Never let analytics failures affect the user.
        return
