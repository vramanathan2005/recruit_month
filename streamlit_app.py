#!/usr/bin/env python3
"""Streamlit dashboard for the Texas commitment-strength board."""

from __future__ import annotations

import html
import math
import re
from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn.metrics import roc_auc_score


DATA_DIR = Path("data")
BOARD_DIR = DATA_DIR / "flip_boards"


REASON_DISPLAY = {
    "geographic fit": "Texas-area recruit",
    "regional proximity": "within reasonable recruiting footprint",
    "regional/rival school": "committed to a Texas rival/regional school",
    "active after commitment": "other schools are still involved",
    "no clear Texas-specific signal captured": "no public Texas signal captured yet",
    "Texas recruit": "Texas high school recruit",
    "regional": "within reasonable recruiting footprint",
    "nearby": "within 250 miles",
    "in state": "in-state",
    "regional/rival commit": "committed to a Texas rival/regional school",
    "other-school OV after commit": "took an OV elsewhere after committing",
    "other-school visit after commit": "visited another school after committing",
    "other-school offers after commit": "picked up offers after committing",
    "visited committed school": "returned to committed school",
    "very early commit": "made an early commitment",
    "early commit": "made an early commitment",
    "first commitment": "first-time commitment",
    "prior decommit": "has decommitted before",
    "only commit at position": "only commit at his position",
    "thin position group": "thin position group at committed school",
    "far": "far from home",
    "cross country": "cross-country commit",
    "4-star": "4-star prospect",
    "5-star": "5-star prospect",
}


st.set_page_config(
    page_title="Texas Flip Board",
    page_icon="TX",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --burnt: #BF5700;
            --burnt-dark: #A04400;
            --black: #111111;
            --ink: #1C1C1C;
            --muted: #6E6E6E;
            --paper: #FFFFFF;
            --line: #D8D8D8;
            --soft: #F4F1ED;
        }

        .stApp {
            background: var(--soft);
            color: var(--ink);
        }

        .block-container {
            padding-top: 0;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] * {
            color: var(--ink);
        }

        [data-testid="stSidebar"] h3 {
            color: var(--black);
            font-size: 0.95rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border-bottom: 3px solid var(--burnt);
            padding-bottom: 10px;
            margin-bottom: 16px;
        }

        .site-shell {
            margin: 0 0 24px;
            border: 1px solid var(--line);
            background: #FFFFFF;
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.06);
        }

        .page-masthead {
            padding: 24px 28px 22px;
            background: #FFFFFF;
        }

        .eyebrow {
            color: var(--burnt);
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 9px;
        }

        .page-title {
            color: var(--black);
            font-size: 2.45rem;
            line-height: 1.05;
            font-weight: 950;
            margin: 0;
            letter-spacing: 0;
        }

        .page-rule {
            width: 72px;
            height: 5px;
            background: var(--burnt);
            margin: 18px 0 14px;
        }

        .page-copy {
            color: #444444;
            max-width: 930px;
            font-size: 1rem;
            line-height: 1.55;
            margin: 0;
        }

        .section-title {
            display: flex;
            align-items: center;
            gap: 12px;
            color: var(--black);
            font-size: 1.15rem;
            font-weight: 950;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin: 22px 0 12px;
        }

        .section-title::before {
            content: "";
            display: inline-block;
            width: 42px;
            height: 4px;
            border-radius: 0;
            background: var(--burnt);
        }

        .briefing-box {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-left: 5px solid var(--burnt);
            padding: 14px 16px;
            margin: 8px 0 16px;
            color: var(--ink);
            font-size: 0.94rem;
            line-height: 1.45;
        }

        .briefing-box strong {
            color: var(--black);
            font-weight: 950;
        }

        [data-testid="stMetric"] {
            background: var(--paper);
            border: 1px solid var(--line);
            border-top: 4px solid var(--burnt);
            border-radius: 0;
            padding: 12px;
            box-shadow: none;
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 700;
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 850;
            font-size: 1.7rem;
            letter-spacing: 0;
            white-space: nowrap;
        }

        [data-testid="stMetricValue"] > div {
            overflow: visible !important;
            text-overflow: clip !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 0;
            padding: 5px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 0;
            color: var(--muted);
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            padding: 10px 16px;
        }

        .stTabs [aria-selected="true"] {
            background: var(--burnt);
            color: #FFFFFF !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 0;
            overflow: hidden;
            box-shadow: none;
        }

        .stAlert {
            border-radius: 0;
            border-color: var(--line);
        }

        a {
            color: var(--burnt-dark);
            font-weight: 700;
        }

        button[kind="primary"],
        .stButton button {
            border-radius: 0;
        }

        .call-sheet {
            width: 100%;
            border-collapse: collapse;
            background: #FFFFFF;
            border: 1px solid var(--line);
            table-layout: fixed;
            margin-top: 14px;
            overflow: visible;
        }

        .call-sheet th {
            background: #111111;
            color: #FFFFFF;
            font-size: 0.72rem;
            font-weight: 950;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            text-align: left;
            padding: 10px 12px;
        }

        .call-sheet td {
            border-top: 1px solid var(--line);
            vertical-align: middle;
            padding: 10px 12px;
            color: var(--ink);
            font-size: 0.88rem;
            line-height: 1.35;
            overflow: visible;
        }

        .call-row {
            position: relative;
        }

        .call-row:hover {
            background: #FBF8F4;
        }

        .call-rank {
            width: 46px;
            color: var(--muted);
            font-weight: 950;
            text-align: right;
        }

        .call-player {
            font-weight: 950;
            color: var(--black);
            margin-bottom: 3px;
        }

        .call-muted {
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 700;
        }

        .meta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 4px;
        }

        .meta-chip {
            display: inline-flex;
            align-items: center;
            background: #F4F1ED;
            border: 1px solid #E2DDD6;
            color: #5E5E5E;
            font-size: 0.7rem;
            font-weight: 850;
            line-height: 1;
            padding: 4px 6px;
            white-space: nowrap;
        }

        .commit-subline {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 750;
            margin-top: 3px;
        }

        .call-risk {
            display: inline-block;
            background: var(--burnt);
            color: #FFFFFF;
            font-size: 0.72rem;
            font-weight: 950;
            text-transform: uppercase;
            padding: 4px 7px;
            margin-bottom: 6px;
        }

        .call-risk.high {
            background: #8F3F00;
        }

        .call-risk.medium {
            background: #555555;
        }

        .call-risk.low {
            background: #2F6B4F;
        }

        .opp-badge {
            display: inline-block;
            background: #111111;
            color: #FFFFFF;
            font-size: 0.72rem;
            font-weight: 950;
            text-transform: uppercase;
            padding: 4px 7px;
            margin-bottom: 5px;
        }

        .opp-badge.attack {
            background: var(--burnt);
        }

        .opp-badge.strong {
            background: #8F3F00;
        }

        .opp-badge.monitor {
            background: #555555;
        }

        .opp-badge.review {
            background: #8F3F00;
        }

        .opp-badge.intel {
            background: #555555;
        }

        .opp-badge.watch {
            background: #777777;
        }

        .opp-badge.no {
            background: #2F2F2F;
        }

        .call-link {
            color: var(--burnt-dark);
            font-weight: 950;
            text-transform: uppercase;
            font-size: 0.74rem;
            text-decoration: none;
        }

        .player-cell {
            position: relative;
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 0;
        }

        .player-avatar,
        .player-avatar-fallback {
            width: 38px;
            height: 46px;
            border: 1px solid var(--line);
            object-fit: cover;
            background: #EFE9E2;
            flex: 0 0 auto;
        }

        .player-avatar-fallback {
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--burnt);
            font-size: 0.72rem;
            font-weight: 950;
        }

        .player-text {
            min-width: 0;
        }

        .profile-popover {
            display: none;
            position: absolute;
            left: 52px;
            top: calc(100% + 6px);
            width: 350px;
            max-height: 330px;
            overflow-y: auto;
            z-index: 30;
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-top: 4px solid var(--burnt);
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
            padding: 11px;
        }

        .player-cell:hover .profile-popover {
            display: block;
        }

        .profile-head {
            display: grid;
            grid-template-columns: 54px 1fr;
            gap: 10px;
            align-items: start;
            margin-bottom: 9px;
        }

        .profile-head img,
        .profile-head .profile-photo-fallback {
            width: 54px;
            height: 68px;
            object-fit: cover;
            border: 1px solid var(--line);
            background: #EFE9E2;
        }

        .profile-photo-fallback {
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--burnt);
            font-size: 0.78rem;
            font-weight: 950;
        }

        .profile-title {
            color: var(--black);
            font-size: 0.98rem;
            font-weight: 950;
            line-height: 1.2;
            margin-bottom: 3px;
        }

        .profile-line {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 750;
            margin-bottom: 2px;
        }

        .profile-grid {
            display: grid;
            grid-template-columns: 62px 1fr;
            gap: 6px 9px;
            border-top: 1px solid var(--line);
            padding-top: 9px;
        }

        .profile-grid dt {
            color: var(--muted);
            font-size: 0.66rem;
            font-weight: 950;
            text-transform: uppercase;
        }

        .profile-grid dd {
            color: var(--ink);
            font-size: 0.76rem;
            line-height: 1.28;
            margin: 0;
        }

        .viz-grid {
            display: grid;
            grid-template-columns: 1.45fr 0.95fr;
            gap: 16px;
            align-items: stretch;
            margin: 12px 0 18px;
        }

        .viz-panel {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-top: 4px solid var(--burnt);
            padding: 14px;
            min-height: 100%;
        }

        .viz-title {
            color: var(--black);
            font-size: 0.86rem;
            font-weight: 950;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .viz-note {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.4;
            margin-bottom: 10px;
        }

        .quality-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 8px 0 10px;
        }

        .quality-chip {
            display: inline-flex;
            align-items: center;
            border: 1px solid #E2DDD6;
            background: #F8F6F2;
            color: #555555;
            font-size: 0.7rem;
            font-weight: 900;
            line-height: 1;
            padding: 5px 7px;
            text-transform: uppercase;
        }

        .quality-chip.good {
            background: #EEF6F1;
            color: #245E42;
            border-color: #C9E1D1;
        }

        .quality-chip.warn {
            background: #FFF6EC;
            color: #8F3F00;
            border-color: #F0D0AE;
        }

        .timeline-wrap {
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding: 8px 0 4px;
            margin: 10px 0 18px;
        }

        .timeline-item {
            min-width: 156px;
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-top: 4px solid #777777;
            padding: 9px 10px;
        }

        .timeline-item.commitment {
            border-top-color: var(--burnt);
        }

        .timeline-item.decommit {
            border-top-color: #111111;
        }

        .timeline-item.visit {
            border-top-color: #6E6E6E;
        }

        .timeline-date {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 900;
            text-transform: uppercase;
            margin-bottom: 5px;
        }

        .timeline-type {
            color: var(--black);
            font-size: 0.82rem;
            font-weight: 950;
            margin-bottom: 4px;
        }

        .timeline-school {
            color: var(--ink);
            font-size: 0.78rem;
            line-height: 1.25;
            font-weight: 700;
        }

        @media (max-width: 980px) {
            .viz-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="site-shell">
            <div class="page-masthead">
                <div class="eyebrow">Texas Football Recruiting</div>
                <h1 class="page-title">Flip Risk Board</h1>
                <div class="page-rule"></div>
                <p class="page-copy">
                    247Sports-based board for finding vulnerable commitments, surfacing
                    Texas-relevant context, and turning public recruiting signals into
                    a staff action list.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def briefing_box(title: str, text: str) -> None:
    st.markdown(
        f'<div class="briefing-box"><strong>{html.escape(title)}</strong><br>{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False).fillna("")


def percent_to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.rstrip("%"), errors="coerce")


def number_to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def player_initials(name: object) -> str:
    parts = [part for part in str(name or "").replace(".", "").split() if part]
    return "".join(part[0].upper() for part in parts[:2]) or "247"


@st.cache_data(show_spinner=False)
def load_player_image_index() -> dict[str, str]:
    path = DATA_DIR / "player_images.csv"
    if not path.exists():
        return {}
    images = pd.read_csv(path, dtype=str).fillna("")
    return dict(zip(images["player_id"], images["image_url"]))


def player_image_url(profile_url: object) -> str:
    match = re.search(r"-(\d+)(?:/|$)", str(profile_url or ""))
    return load_player_image_index().get(match.group(1), "") if match else ""


def avatar_html(image_url: str, name: object, large: bool = False) -> str:
    initials = html.escape(player_initials(name))
    if image_url:
        class_name = "profile-photo" if large else "player-avatar"
        return f'<img class="{class_name}" src="{html.escape(image_url)}" alt="{html.escape(str(name or ""))}">'
    class_name = "profile-photo-fallback" if large else "player-avatar-fallback"
    return f'<div class="{class_name}">{initials}</div>'


def risk_label(probability: object) -> str:
    try:
        value = float(probability)
    except (TypeError, ValueError):
        return "Unknown"
    if value >= 70:
        return "Very High"
    if value >= 50:
        return "High"
    if value >= 35:
        return "Medium"
    return "Low"


def clean_phrase(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "na", "none"}:
        return ""
    return text.replace("_", " ")


def clean_rank(value: object) -> str:
    text = clean_phrase(value)
    if not text:
        return "Unranked"
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def clean_notes(value: object) -> str:
    text = clean_phrase(value)
    return text if text else "None captured"


def display_reason(reason: object) -> str:
    text = clean_phrase(reason)
    return REASON_DISPLAY.get(text, text)


def display_reason_list(value: object, fallback: str = "No reason captured") -> str:
    text = clean_phrase(value)
    if not text:
        return fallback
    reasons = [display_reason(part.strip()) for part in text.split(";") if part.strip()]
    return "; ".join(dict.fromkeys(reasons)) if reasons else fallback


def is_yes(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "x"}


def activity_summary(row: pd.Series) -> str:
    pieces = []
    official_visits = int(pd.to_numeric(row.get("post_commit_other_school_official_visits", 0), errors="coerce") or 0)
    other_visits = int(pd.to_numeric(row.get("post_commit_other_school_visits", 0), errors="coerce") or 0)
    offers = int(pd.to_numeric(row.get("post_commit_other_school_offers", 0), errors="coerce") or 0)
    if official_visits:
        pieces.append(f"{official_visits} OV elsewhere")
    if other_visits:
        pieces.append(f"{other_visits} total visits elsewhere")
    if offers:
        pieces.append(f"{offers} new offers")
    return "; ".join(pieces) if pieces else "No outside activity captured"


def miles_summary(value: object, label: str) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return ""
    return f"{int(round(float(number))):,} mi {label}"


def home_label(row: pd.Series) -> str:
    city = clean_phrase(row.get("home_city", ""))
    state = clean_phrase(row.get("home_state", ""))
    if city and state:
        return f"{city}, {state}"
    return state


def target_distance_summary(row: pd.Series) -> str:
    distance = miles_summary(row.get("home_to_target_miles", ""), "to Austin")
    if distance:
        return distance
    if str(row.get("target_distance_bucket", "")).strip() == "in_state":
        return "Texas HS"
    return ""


def staff_summary(row: pd.Series) -> str:
    pieces = []
    if is_yes(row.get("staff_priority")):
        pieces.append("staff priority")
    if is_yes(row.get("texas_offer")):
        pieces.append("Texas offer")
    if is_yes(row.get("texas_visit")):
        pieces.append("Texas visit")
    if is_yes(row.get("texas_recruiting")):
        pieces.append("Texas recruiting")
    notes = clean_phrase(row.get("staff_notes", ""))
    if notes:
        pieces.append(notes)
    return "; ".join(pieces) if pieces else "No staff override captured"


def texas_angle(row: pd.Series) -> str:
    pieces = []
    distance = target_distance_summary(row)
    if distance:
        pieces.append(distance)
    notes = clean_phrase(row.get("texas_context_tags", ""))
    if notes:
        pieces.extend(display_reason(note.strip()) for note in notes.split(";") if note.strip())
    return "; ".join(pieces) if pieces else "No special Texas angle captured"


def readable_reasons(row: pd.Series) -> str:
    reasons = []
    raw = str(row.get("why_flagged", "") or "")
    for reason in [part.strip() for part in raw.split(";") if part.strip()]:
        reasons.append(display_reason(reason))
    return "; ".join(reasons[:5])


def numeric_value(value: object) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return 0.0 if pd.isna(number) else float(number)


def risk_read(row: pd.Series) -> str:
    label = risk_label(row.get("model_decommit_flip_probability_num", ""))
    loose_signals = readable_reasons(row).split("; ") if readable_reasons(row) else []
    other_activity = sum(
        numeric_value(row.get(field, 0))
        for field in [
            "post_commit_other_school_official_visits",
            "post_commit_other_school_visits",
            "post_commit_other_school_offers",
        ]
    )
    committed_school_visits = numeric_value(row.get("post_commit_committed_school_visits", 0))
    stick = numeric_value(row.get("model_stick_probability_num", 0))

    stable_signals = []
    if not other_activity:
        stable_signals.append("no outside activity captured")
    if committed_school_visits:
        stable_signals.append("has been back to committed school")
    if stick >= 65:
        stable_signals.append(f"{stick:.1f}% stick chance")

    if label in {"Very High", "High"}:
        return "; ".join(loose_signals[:3]) if loose_signals else "model sees several loose-commitment signals"
    if label == "Medium":
        mixed = loose_signals[:2] + stable_signals[:1]
        return "; ".join(mixed) if mixed else "mixed profile; not a clear attack"
    if label == "Low":
        return "; ".join(stable_signals[:3]) if stable_signals else "few loose-commitment signals captured"
    return "not enough signal captured"


def short_risk_read(row: pd.Series) -> str:
    read = risk_read(row)
    return "; ".join(read.split("; ")[:2])


def short_context(row: pd.Series, texas_mode: bool = False) -> str:
    pieces = []
    if texas_mode:
        distance = target_distance_summary(row)
        if distance:
            pieces.append(distance)
        notes = clean_phrase(row.get("texas_context_tags", ""))
        if notes:
            pieces.append(notes.split(";")[0].strip())
    else:
        distance = miles_summary(row.get("home_to_school_miles", ""), "from home")
        if distance:
            pieces.append(distance)
    activity = activity_summary(row)
    if activity != "No outside activity captured":
        pieces.append(activity)
    return "; ".join(pieces) if pieces else "No extra context captured"


def data_quality_notes(row: pd.Series, texas_mode: bool = False) -> list[tuple[str, str]]:
    notes: list[tuple[str, str]] = []
    lat = pd.to_numeric(row.get("home_lat", ""), errors="coerce")
    lon = pd.to_numeric(row.get("home_lon", ""), errors="coerce")
    precision = clean_phrase(row.get("geography_precision", ""))
    if pd.notna(lat) and pd.notna(lon) and precision and "state" not in precision:
        notes.append(("good", "Mapped to city"))
    elif pd.notna(lat) and pd.notna(lon):
        notes.append(("warn", "State-level location"))
    else:
        notes.append(("warn", "Map location missing"))
    if texas_mode and not clean_phrase(row.get("texas_context_tags", "")):
        notes.append(("warn", "No Texas signal captured"))
    if not activity_summary(row) or activity_summary(row) == "No outside activity captured":
        notes.append(("warn", "No outside activity captured"))
    else:
        notes.append(("good", "Outside activity captured"))
    return notes


def data_quality_html(row: pd.Series, texas_mode: bool = False) -> str:
    chips = []
    for status, label in data_quality_notes(row, texas_mode=texas_mode):
        chips.append(f'<span class="quality-chip {html.escape(status)}">{html.escape(label)}</span>')
    return '<div class="quality-row">' + "".join(chips) + '</div>'


def data_quality_text(row: pd.Series, texas_mode: bool = False) -> str:
    return "; ".join(label for _, label in data_quality_notes(row, texas_mode=texas_mode))


def prepare_board(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "board_rank",
        "live_commitment_strength_score",
        "commitment_strength_score",
        "national_rank",
        "home_lat",
        "home_lon",
        "college_lat",
        "college_lon",
        "target_lat",
        "target_lon",
        "home_to_target_miles",
        "home_to_school_miles",
        "post_commit_other_school_official_visits",
        "post_commit_other_school_visits",
        "post_commit_other_school_offers",
        "post_commit_committed_school_visits",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = number_to_float(df[col])
    for col in [
        "model_decommit_flip_probability",
        "model_stick_probability",
        "estimated_decommit_flip_probability",
        "estimated_stick_probability",
        "historical_bucket_decommit_flip_rate",
    ]:
        if col in df.columns:
            df[col + "_num"] = percent_to_float(df[col])
    return df


def model_year_history(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    df["class_year"] = pd.to_numeric(df["class_year"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["model_decommit_flip_probability"] = pd.to_numeric(
        df["model_decommit_flip_probability"],
        errors="coerce",
    )
    df = df.dropna(subset=["class_year", "target", "model_decommit_flip_probability"])
    df = df[df["class_year"] <= 2026]
    rows = []
    for year, group in df.groupby("class_year"):
        positives = int(group["target"].sum())
        total = len(group)
        predicted_high = group["model_decommit_flip_probability"] >= 0.40
        actual_high = group["target"] == 1
        high_risk_count = int(predicted_high.sum())
        high_risk_hits = int((predicted_high & actual_high).sum())
        top_20 = group.sort_values("model_decommit_flip_probability", ascending=False).head(20)
        top_20_hits = int(top_20["target"].sum())
        row = {
            "Class": int(year),
            "Known Commits": total,
            "Actually Flipped/Decommitted": positives,
            "Actual Rate": f"{positives / total:.1%}" if total else "",
            "Avg Predicted Risk": f"{group['model_decommit_flip_probability'].mean():.1%}" if total else "",
            "High-Risk Hits": f"{high_risk_hits}/{high_risk_count}" if high_risk_count else "0/0",
            "Top 20 Hits": f"{top_20_hits}/20" if len(top_20) == 20 else f"{top_20_hits}/{len(top_20)}",
        }
        if positives and positives < total:
            row["Separation Score"] = f"{roc_auc_score(group['target'], group['model_decommit_flip_probability']):.1%}"
        else:
            row["Separation Score"] = "n/a"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Class")


def model_year_bucket_history(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    df["class_year"] = pd.to_numeric(df["class_year"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["model_decommit_flip_probability"] = pd.to_numeric(
        df["model_decommit_flip_probability"],
        errors="coerce",
    )
    df = df.dropna(subset=["class_year", "target", "model_decommit_flip_probability"])
    df = df[df["class_year"] <= 2026]
    df["Risk Bucket"] = pd.cut(
        df["model_decommit_flip_probability"],
        bins=[0, 0.10, 0.20, 0.30, 0.40, 1.0],
        labels=["0-10%", "10-20%", "20-30%", "30-40%", "40%+"],
        include_lowest=True,
    )
    rows = []
    for (year, bucket), group in df.groupby(["class_year", "Risk Bucket"], observed=True):
        hits = int(group["target"].sum())
        total = len(group)
        rows.append(
            {
                "Class": int(year),
                "Risk Bucket": str(bucket),
                "Players": total,
                "Flipped/Decommitted": hits,
                "Actual Rate": f"{hits / total:.1%}" if total else "",
                "Avg Predicted Risk": f"{group['model_decommit_flip_probability'].mean():.1%}" if total else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["Class", "Risk Bucket"])


def rolling_chart_data(rolling: pd.DataFrame) -> pd.DataFrame:
    chart = pd.DataFrame()
    chart["Test Class"] = rolling["test_year"].astype(str)
    chart["Actual Flip Rate"] = rolling["actual_decommit_flip_rate"].map(lambda value: round(float(value) * 100, 1))
    chart["Avg Board Risk"] = rolling["average_predicted_probability"].map(lambda value: round(float(value) * 100, 1))
    chart["Separated Movers From Stayers"] = rolling["auc"].map(lambda value: round(float(value) * 100, 1))
    chart["Top 20 That Moved"] = rolling["top_20_hit_rate"].map(lambda value: round(float(value) * 100, 1))
    return chart


def next_school_after(row: pd.Series, timeline_events: pd.DataFrame | None) -> str:
    if timeline_events is None or timeline_events.empty:
        return ""
    outcome_date = pd.to_datetime(row.get("outcome_date", ""), errors="coerce")
    if pd.isna(outcome_date):
        return ""
    player_events = timeline_events[
        (timeline_events["class_year"].astype(str) == str(int(row["class_year"])))
        & (timeline_events["player_id"].astype(str) == str(row.get("player_id", "")))
    ].copy()
    if player_events.empty:
        return ""
    player_events["event_dt"] = pd.to_datetime(player_events["event_date"], errors="coerce")
    future = player_events[
        (player_events["event_dt"] > outcome_date)
        & (player_events["event_type"].isin(["Commitment", "Signing", "Enrollment"]))
    ].sort_values("event_dt")
    if future.empty:
        return ""
    return str(future.iloc[0].get("school", ""))


def timeline_groups(timeline_events: pd.DataFrame | None) -> dict[tuple[str, str], pd.DataFrame]:
    if timeline_events is None or timeline_events.empty:
        return {}
    grouped = {}
    events = timeline_events.copy()
    events["event_dt"] = pd.to_datetime(events["event_date"], errors="coerce")
    for key, group in events.groupby([events["class_year"].astype(str), events["player_id"].astype(str)]):
        grouped[key] = group.sort_values("event_dt")
    return grouped


def next_school_after_from_groups(row: pd.Series, groups: dict[tuple[str, str], pd.DataFrame]) -> str:
    outcome_date = pd.to_datetime(row.get("outcome_date", ""), errors="coerce")
    if pd.isna(outcome_date):
        return ""
    key = (str(int(row["class_year"])), str(row.get("player_id", "")))
    player_events = groups.get(key)
    if player_events is None or player_events.empty:
        return ""
    future = player_events[
        (player_events["event_dt"] > outcome_date)
        & (player_events["event_type"].isin(["Commitment", "Signing", "Enrollment"]))
    ].sort_values("event_dt")
    if future.empty:
        return ""
    return str(future.iloc[0].get("school", ""))


def outcome_summary(row: pd.Series, timeline_events: pd.DataFrame | None) -> tuple[str, str]:
    outcome = str(row.get("outcome", ""))
    school = str(row.get("outcome_school", ""))
    committed_school = str(row.get("committed_school_at_event", ""))
    if outcome == "decommitted":
        next_school = next_school_after(row, timeline_events)
        if next_school and clean_team_name(next_school) == clean_team_name(committed_school):
            return f"Decommitted from {school}, then recommitted", next_school
        return f"Decommitted from {school}", next_school or "No later school captured"
    if outcome == "flipped":
        return f"Flipped to {school}", school
    if outcome == "stuck":
        return f"Stuck with {school}", school
    return outcome or "Unknown", ""


def high_school_cutoff_for_year(class_year: object) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(year=int(float(class_year)), month=8, day=1)
    except (TypeError, ValueError):
        return None


def timeline_for_player(row: pd.Series, timeline_events: pd.DataFrame | None) -> pd.DataFrame:
    if timeline_events is None or timeline_events.empty:
        return pd.DataFrame()
    class_year = str(int(float(row.get("class_year", 0)))) if clean_phrase(row.get("class_year", "")) else ""
    player_id = clean_phrase(row.get("player_id", ""))
    if not class_year or not player_id:
        return pd.DataFrame()
    events = timeline_events[
        (timeline_events["class_year"].astype(str) == class_year)
        & (timeline_events["player_id"].astype(str) == player_id)
    ].copy()
    if events.empty:
        return events
    events["event_dt"] = pd.to_datetime(events["event_date"], errors="coerce")
    cutoff = high_school_cutoff_for_year(class_year)
    if cutoff is not None:
        events = events[(events["event_dt"].isna()) | (events["event_dt"] <= cutoff)]
    return events.sort_values("event_dt").drop_duplicates(
        subset=["event_type", "event_date", "school"],
        keep="first",
    )


def timeline_event_class(event_type: object) -> str:
    text = clean_phrase(event_type).lower()
    if "commitment" in text or "signing" in text or "enrollment" in text:
        return "commitment"
    if "decommit" in text:
        return "decommit"
    if "visit" in text:
        return "visit"
    return ""


def display_player_timeline(df: pd.DataFrame, timeline_events: pd.DataFrame, key: str) -> None:
    if df.empty:
        st.info("No players in the current filters.")
        return
    options = df.apply(
        lambda row: f"{row.get('name', '')} | {row.get('position', '')} | {row.get('committed_team', '')}",
        axis=1,
    ).tolist()
    selected = st.selectbox("Player timeline", options, key=key)
    selected_index = options.index(selected)
    row = df.iloc[selected_index]
    events = timeline_for_player(row, timeline_events)
    st.markdown(data_quality_html(row, texas_mode="texas" in key), unsafe_allow_html=True)
    if events.empty:
        st.info("No timeline events captured for this player.")
        return
    items = []
    for _, event in events.tail(18).iterrows():
        event_type = html.escape(clean_phrase(event.get("event_type", "")))
        school = html.escape(clean_phrase(event.get("school", "")) or "School not captured")
        event_date = html.escape(clean_phrase(event.get("event_date", "")))
        item_class = timeline_event_class(event.get("event_type", ""))
        items.append(
            f'<div class="timeline-item {item_class}">'
            f'<div class="timeline-date">{event_date}</div>'
            f'<div class="timeline-type">{event_type}</div>'
            f'<div class="timeline-school">{school}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="timeline-wrap">{"".join(items)}</div>', unsafe_allow_html=True)


def clean_team_name(value: object) -> str:
    text = str(value or "").lower()
    for suffix in [
        " hokies", " ducks", " gators", " bulldogs", " tigers", " wildcats",
        " longhorns", " aggies", " sooners", " buckeyes", " trojans",
        " seminoles", " hurricanes", " volunteers", " tar heels", " rebels",
        " badgers", " nittany lions", " spartans", " bears", " horned frogs",
        " red raiders", " knights", " sun devils", " buffaloes", " cardinals",
        " fighting irish", " terrapins", " boilermakers", " commodores",
        " yellow jackets", " wolfpack", " panthers", " orange", " cavaliers",
        " demon deacons",
    ]:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return re.sub(r"[^a-z0-9]+", "", text)


def recommitted_same_school(row: pd.Series, timeline_events: pd.DataFrame | None) -> bool:
    if str(row.get("outcome", "")) != "decommitted":
        return False
    next_school = next_school_after(row, timeline_events)
    return bool(next_school and clean_team_name(next_school) == clean_team_name(row.get("committed_school_at_event", "")))


def recommitted_same_school_from_groups(row: pd.Series, groups: dict[tuple[str, str], pd.DataFrame]) -> bool:
    if str(row.get("outcome", "")) != "decommitted":
        return False
    next_school = next_school_after_from_groups(row, groups)
    return bool(next_school and clean_team_name(next_school) == clean_team_name(row.get("committed_school_at_event", "")))


def historical_player_examples(
    events: pd.DataFrame,
    timeline_events: pd.DataFrame | None = None,
    year: int | None = None,
    limit: int = 40,
) -> pd.DataFrame:
    df = events.copy()
    df["class_year"] = pd.to_numeric(df["class_year"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["model_decommit_flip_probability"] = pd.to_numeric(
        df["model_decommit_flip_probability"],
        errors="coerce",
    )
    df = df.dropna(subset=["class_year", "target", "model_decommit_flip_probability"])
    df = df[df["class_year"] <= 2026]
    if year:
        df = df[df["class_year"] == year]
    groups = timeline_groups(timeline_events)
    df = (
        df.sort_values("model_decommit_flip_probability", ascending=False)
        .drop_duplicates(subset=["player_id"], keep="first")
        .head(limit)
    )
    rows = []
    for _, row in df.iterrows():
        predicted_risky = row["model_decommit_flip_probability"] >= 0.40
        actually_moved = row["target"] == 1
        if predicted_risky and actually_moved:
            result = "Flagged correctly"
        elif predicted_risky:
            result = "Flagged but stuck"
        elif actually_moved:
            result = "Missed"
        else:
            result = "Correctly stable"
        what_happened, next_school = outcome_summary(row, timeline_events)
        rows.append(
            {
                "Class": int(row["class_year"]),
                "Sample": "Training" if int(row["class_year"]) <= 2024 else "Holdout/Out-of-sample",
                "Player": row.get("name", ""),
                "Pos": row.get("position", ""),
                "Committed To": row.get("committed_school_at_event", ""),
                "Model Risk": f"{row['model_decommit_flip_probability']:.1%}",
                "What Happened": what_happened,
                "Next School": next_school,
                "Result": result,
            }
        )
    return pd.DataFrame(rows)


def model_review_examples(
    events: pd.DataFrame,
    timeline_events: pd.DataFrame | None = None,
    year: int | None = None,
    category: str = "Moved but missed",
    limit: int = 40,
) -> pd.DataFrame:
    df = events.copy()
    df["class_year"] = pd.to_numeric(df["class_year"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["model_decommit_flip_probability"] = pd.to_numeric(
        df["model_decommit_flip_probability"],
        errors="coerce",
    )
    df = df.dropna(subset=["class_year", "target", "model_decommit_flip_probability"])
    df = df[df["class_year"] <= 2026]
    if year:
        df = df[df["class_year"] == year]
    groups = timeline_groups(timeline_events)

    predicted_risky = df["model_decommit_flip_probability"] >= 0.40
    actually_moved = df["target"] == 1
    masks = {
        "Correctly flagged": predicted_risky & actually_moved,
        "Flagged but stayed": predicted_risky & ~actually_moved,
        "Moved but missed": ~predicted_risky & actually_moved,
        "Ended elsewhere but missed": ~predicted_risky & actually_moved,
        "Temporary decommit/recommit": actually_moved,
        "Correctly stable": ~predicted_risky & ~actually_moved,
    }
    df = df[masks.get(category, masks["Moved but missed"])].copy()
    if category in {"Moved but missed", "Ended elsewhere but missed"}:
        df = df[~df.apply(lambda row: recommitted_same_school_from_groups(row, groups), axis=1)]
    elif category == "Temporary decommit/recommit":
        df = df[df.apply(lambda row: recommitted_same_school_from_groups(row, groups), axis=1)]

    if category in {"Correctly flagged", "Flagged but stayed"}:
        df = df.sort_values("model_decommit_flip_probability", ascending=False)
    elif category == "Moved but missed":
        df = df.sort_values("model_decommit_flip_probability", ascending=True)
    else:
        df = df.sort_values("model_decommit_flip_probability", ascending=True)

    df = df.drop_duplicates(subset=["player_id"], keep="first").head(limit)
    rows = []
    for _, row in df.iterrows():
        what_happened, next_school = outcome_summary(row, timeline_events)
        rows.append(
            {
                "Class": int(row["class_year"]),
                "Player": row.get("name", ""),
                "Pos": row.get("position", ""),
                "Committed To": row.get("committed_school_at_event", ""),
                "Board Risk": f"{row['model_decommit_flip_probability']:.1%}",
                "What Happened": what_happened,
                "Next School": next_school,
                "Why Staff Should Care": {
                    "Correctly flagged": "Board found a loose commit.",
                    "Flagged but stayed": "Board was too aggressive here.",
                    "Moved but missed": "Board under-rated a true loss.",
                    "Ended elsewhere but missed": "Board under-rated a true loss.",
                    "Temporary decommit/recommit": "Player wobbled but ended at same school.",
                    "Correctly stable": "Board treated this as stable.",
                }[category],
            }
        )
    return pd.DataFrame(rows)


def risk_class(row: pd.Series) -> str:
    label = risk_label(row.get("model_decommit_flip_probability_num", ""))
    if label == "High":
        return "high"
    if label == "Medium":
        return "medium"
    if label == "Low":
        return "low"
    return ""


def map_color(row: pd.Series, texas_mode: bool = False) -> list[int]:
    risk = numeric_value(row.get("model_decommit_flip_probability_num", 0))
    if risk >= 70:
        return [191, 87, 0, 225]
    if risk >= 50:
        return [143, 63, 0, 210]
    if risk >= 35:
        return [85, 85, 85, 190]
    return [47, 107, 79, 170]


def spread_overlapping_hometowns(mapped: pd.DataFrame) -> pd.DataFrame:
    """Spread exact same-city points just enough to make each prospect hoverable."""
    if mapped.empty:
        return mapped
    out = mapped.copy()
    out["cluster_key"] = out["lat"].round(4).astype(str) + "," + out["lon"].round(4).astype(str)
    out["cluster_count"] = out.groupby("cluster_key")["cluster_key"].transform("size")
    out["cluster_index"] = out.groupby("cluster_key").cumcount()
    out["raw_lat"] = out["lat"]
    out["raw_lon"] = out["lon"]

    def spread(row: pd.Series) -> tuple[float, float]:
        count = int(row["cluster_count"])
        if count <= 1:
            return float(row["lat"]), float(row["lon"])
        index = int(row["cluster_index"])
        ring = index // 8 + 1
        slot = index % 8
        angle = (2 * math.pi * slot / min(count, 8)) + (ring * 0.35)
        radius = 0.12 * ring
        lat_offset = math.cos(angle) * radius
        lon_scale = max(math.cos(math.radians(float(row["lat"]))), 0.25)
        lon_offset = math.sin(angle) * radius / lon_scale
        return float(row["lat"]) + lat_offset, float(row["lon"]) + lon_offset

    spread_points = out.apply(spread, axis=1, result_type="expand")
    out["lat"] = spread_points[0]
    out["lon"] = spread_points[1]
    out["HomeCluster"] = out.apply(
        lambda row: f"{int(row['cluster_count'])} prospects near {row['Home']}"
        if int(row["cluster_count"]) > 1
        else "single prospect hometown",
        axis=1,
    )
    return out


def map_rows(df: pd.DataFrame, texas_mode: bool = False) -> pd.DataFrame:
    if df.empty or "home_lat" not in df.columns or "home_lon" not in df.columns:
        return pd.DataFrame()
    mapped = df.copy()
    mapped["lat"] = pd.to_numeric(mapped["home_lat"], errors="coerce")
    mapped["lon"] = pd.to_numeric(mapped["home_lon"], errors="coerce")
    mapped = mapped.dropna(subset=["lat", "lon"]).copy()
    if mapped.empty:
        return mapped
    mapped["Risk"] = mapped["model_decommit_flip_probability_num"].map(lambda value: f"{float(value):.1f}%" if pd.notna(value) else "n/a")
    mapped["Rank"] = mapped["national_rank"].map(clean_rank)
    mapped["Home"] = mapped.apply(home_label, axis=1)
    mapped["Action"] = mapped["model_decommit_flip_probability_num"].map(risk_label)
    dist_col = "target_distance_bucket" if texas_mode else "distance_bucket"
    mapped["Fit"] = mapped[dist_col].map(clean_phrase) if dist_col in mapped.columns else ""
    mapped["Reason"] = mapped.apply(lambda row: short_risk_read(row), axis=1)
    mapped["Context"] = mapped.apply(lambda row: short_context(row, texas_mode=texas_mode), axis=1)
    mapped["Photo"] = mapped["profile_url"].map(player_image_url) if "profile_url" in mapped.columns else ""
    mapped["PhotoTag"] = mapped["Photo"].map(
        lambda url: f'<img src="{html.escape(str(url))}" style="width:48px;height:60px;object-fit:cover;border:1px solid #D8D8D8;background:#F4F1ED;">'
        if clean_phrase(url)
        else '<div style="width:48px;height:60px;border:1px solid #D8D8D8;background:#F4F1ED;"></div>'
    )
    mapped["ActionColor"] = mapped["model_decommit_flip_probability_num"].map(
        lambda v: "#BF5700" if numeric_value(v) >= 50 else "#555555"
    )
    mapped["color"] = mapped.apply(lambda row: map_color(row, texas_mode=texas_mode), axis=1)
    mapped["radius"] = mapped["model_decommit_flip_probability_num"].fillna(20).clip(20, 100).map(lambda value: 5 + float(value) / 14)
    return spread_overlapping_hometowns(mapped)


def school_map_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "college_lat" not in df.columns or "college_lon" not in df.columns:
        return pd.DataFrame()
    mapped = df.copy()
    mapped["lat"] = pd.to_numeric(mapped["college_lat"], errors="coerce")
    mapped["lon"] = pd.to_numeric(mapped["college_lon"], errors="coerce")
    mapped = mapped.dropna(subset=["lat", "lon"])
    if mapped.empty:
        return mapped
    grouped = mapped.groupby(["committed_team", "lat", "lon"], dropna=False).agg(
        Players=("name", "count"),
        AvgRisk=("model_decommit_flip_probability_num", "mean"),
    ).reset_index()
    grouped["Risk"] = grouped["AvgRisk"].map(lambda value: f"{float(value):.1f}%" if pd.notna(value) else "n/a")
    grouped["radius"] = grouped["Players"].clip(1, 35).map(lambda value: 7 + int(value) * 0.8)
    grouped["color"] = grouped["AvgRisk"].fillna(0).map(lambda risk: [191, 87, 0, 220] if risk >= 50 else [85, 85, 85, 185])
    return grouped


def austin_ring_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "target_lat" not in df.columns or "target_lon" not in df.columns:
        return pd.DataFrame()
    target = df.dropna(subset=["target_lat", "target_lon"])
    if target.empty:
        return pd.DataFrame()
    lat = float(target.iloc[0]["target_lat"])
    lon = float(target.iloc[0]["target_lon"])
    return pd.DataFrame(
        [
            {"lat": lat, "lon": lon, "Miles": "250 mi", "radius": 250 * 1609.34, "color": [191, 87, 0, 35]},
            {"lat": lat, "lon": lon, "Miles": "750 mi", "radius": 750 * 1609.34, "color": [191, 87, 0, 20]},
            {"lat": lat, "lon": lon, "Miles": "1500 mi", "radius": 1500 * 1609.34, "color": [191, 87, 0, 12]},
        ]
    )


def display_recruiting_map(df: pd.DataFrame, texas_mode: bool = False, key: str = "map") -> None:
    home_points = map_rows(df, texas_mode=texas_mode)
    if home_points.empty:
        st.info("No mappable hometown coordinates captured for these filters.")
        return
    center_lat = float(home_points["lat"].mean())
    center_lon = float(home_points["lon"].mean())
    layers = []
    if texas_mode:
        rings = austin_ring_rows(df)
        if not rings.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=rings,
                    get_position="[lon, lat]",
                    get_radius="radius",
                    get_fill_color="color",
                    get_line_color=[191, 87, 0, 90],
                    stroked=True,
                    filled=True,
                    line_width_min_pixels=1,
                )
            )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=home_points,
            get_position="[lon, lat]",
            get_radius="radius",
            radius_units="pixels",
            radius_min_pixels=5,
            radius_max_pixels=13,
            get_fill_color="color",
            get_line_color=[17, 17, 17, 180],
            pickable=True,
            stroked=True,
            filled=True,
            line_width_min_pixels=1,
        )
    )
    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=3.25, pitch=0),
        layers=layers,
        tooltip={
            "html": (
                "<div style='width:310px;background:white;color:#1C1C1C;border-top:4px solid #BF5700;padding:10px;font-family:Arial, sans-serif;'>"
                "<div style='display:flex;gap:10px;align-items:flex-start;margin-bottom:8px;'>"
                "{PhotoTag}"
                "<div>"
                "<div style='font-size:15px;font-weight:900;line-height:1.15;margin-bottom:3px;'>{name}</div>"
                "<div style='font-size:12px;color:#666;font-weight:700;'>{position} | {star_bucket} | #{Rank}</div>"
                "<div style='font-size:12px;color:#666;font-weight:700;margin-top:2px;'>{Home}</div>"
                "</div>"
                "</div>"
                "<div style='display:inline-block;background:{ActionColor};color:white;font-size:11px;font-weight:900;text-transform:uppercase;padding:4px 7px;margin-bottom:7px;'>{Action}</div>"
                "<div style='font-size:13px;margin-bottom:4px;'><b>Committed:</b> {committed_team}</div>"
                "<div style='font-size:13px;margin-bottom:4px;'><b>Move risk:</b> {Risk}</div>"
                "<div style='font-size:13px;margin-bottom:4px;'><b>Fit:</b> {Fit}</div>"
                "<div style='font-size:12px;color:#777;margin-bottom:4px;'>{HomeCluster}</div>"
                "<div style='font-size:12px;color:#555;line-height:1.3;'>{Context}</div>"
                "</div>"
            ),
            "style": {"backgroundColor": "transparent", "color": "#1C1C1C", "fontFamily": "Arial"},
        },
    )
    st.pydeck_chart(deck, use_container_width=True)
    clustered = int((home_points["cluster_count"] > 1).sum()) if "cluster_count" in home_points.columns else 0
    cluster_note = f" {clustered:,} same-city dots are spread slightly so each player can be selected." if clustered else ""
    st.caption(f"{len(home_points):,} of {len(df):,} filtered players have map-ready hometown coordinates.{cluster_note}")


def display_school_map(df: pd.DataFrame, key: str = "school_map") -> None:
    schools = school_map_rows(df)
    if schools.empty:
        st.info("No committed-school coordinates captured for these filters.")
        return
    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=float(schools["lat"].mean()), longitude=float(schools["lon"].mean()), zoom=3.1),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=schools,
                get_position="[lon, lat]",
                get_radius="radius",
                radius_units="pixels",
                radius_min_pixels=7,
                radius_max_pixels=22,
                get_fill_color="color",
                get_line_color=[17, 17, 17, 180],
                pickable=True,
                stroked=True,
                filled=True,
                line_width_min_pixels=1,
            )
        ],
        tooltip={
            "html": "<b>{committed_team}</b><br/>Weak commits: {Players}<br/>Avg risk: {Risk}",
            "style": {"backgroundColor": "#111111", "color": "white", "fontFamily": "Arial"},
        },
    )
    st.pydeck_chart(deck, use_container_width=True)


def top_bar_chart(df: pd.DataFrame, group_col: str, metric_col: str, title: str, limit: int = 10) -> None:
    if df.empty or group_col not in df.columns or metric_col not in df.columns:
        st.info("Not enough data for this chart.")
        return
    chart_df = df.copy()
    chart_df[metric_col] = pd.to_numeric(chart_df[metric_col], errors="coerce")
    chart_df = chart_df.dropna(subset=[group_col, metric_col])
    chart_df = (
        chart_df.groupby(group_col, as_index=False)
        .agg(Players=("name", "count"), AvgRisk=(metric_col, "mean"))
        .sort_values(["Players", "AvgRisk"], ascending=False)
        .head(limit)
    )
    if chart_df.empty:
        st.info("Not enough data for this chart.")
        return
    chart = (
        alt.Chart(chart_df)
        .mark_bar(color="#BF5700")
        .encode(
            x=alt.X("Players:Q", title="Players"),
            y=alt.Y(f"{group_col}:N", title="", sort="-x"),
            tooltip=[group_col, "Players", alt.Tooltip("AvgRisk:Q", format=".1f", title="Avg Risk")],
        )
        .properties(height=max(220, 24 * len(chart_df)), title=title)
    )
    st.altair_chart(chart, use_container_width=True)


def risk_distribution_chart(df: pd.DataFrame) -> None:
    if df.empty or "model_decommit_flip_probability_num" not in df.columns:
        st.info("Not enough risk data for this chart.")
        return
    chart_df = df.copy()
    chart_df["Risk Bucket"] = chart_df["model_decommit_flip_probability_num"].map(risk_label)
    order = ["Very High", "High", "Medium", "Low", "Unknown"]
    counts = chart_df.groupby("Risk Bucket", as_index=False).size()
    counts["Risk Bucket"] = pd.Categorical(counts["Risk Bucket"], categories=order, ordered=True)
    counts = counts.sort_values("Risk Bucket")
    chart = (
        alt.Chart(counts)
        .mark_bar(color="#111111")
        .encode(
            x=alt.X("Risk Bucket:N", sort=order, title="Risk Bucket"),
            y=alt.Y("size:Q", title="Players"),
            tooltip=["Risk Bucket", alt.Tooltip("size:Q", title="Players")],
        )
        .properties(height=220, title="Risk Bucket Mix")
    )
    st.altair_chart(chart, use_container_width=True)


def action_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "staff_action" not in df.columns:
        return pd.DataFrame()
    action_order = ["Call Today", "Coach Review", "Need Intel", "Watch", "Do Not Chase"]
    rows = []
    for action in action_order:
        group = df[df["staff_action"] == action]
        if group.empty:
            continue
        top_positions = ", ".join(group["position"].value_counts().head(3).index.astype(str))
        top_schools = ", ".join(group["committed_team"].value_counts().head(3).index.astype(str))
        rows.append(
            {
                "Staff Action": action,
                "Players": len(group),
                "Avg Move Risk": f"{group['model_decommit_flip_probability_num'].mean():.1f}%",
                "Top Positions": top_positions,
                "Top Schools": top_schools,
                "High Confidence": int((group.get("data_confidence", "") == "High").sum()) if "data_confidence" in group.columns else 0,
            }
        )
    return pd.DataFrame(rows)


def review_queue(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "staff_action" not in work.columns:
        return pd.DataFrame()
    priority = {"Need Intel": 0, "Coach Review": 1, "Call Today": 2, "Watch": 3, "Do Not Chase": 4}
    confidence_rank = {"Low": 0, "Medium": 1, "High": 2}
    work["action_rank"] = work["staff_action"].map(priority).fillna(9)
    if "data_confidence" in work.columns:
        work["confidence_rank"] = work["data_confidence"].map(confidence_rank).fillna(9)
    else:
        work["confidence_rank"] = 9
    work = work.sort_values(
        ["action_rank", "confidence_rank", "model_decommit_flip_probability_num", "board_rank"],
        ascending=[True, True, False, True],
    ).head(limit)
    columns = [
        "staff_action",
        "data_confidence",
        "name",
        "position",
        "committed_team",
        "model_decommit_flip_probability",
        "home_city",
        "home_state",
        "texas_opportunity_reasons",
        "data_notes",
        "next_action",
        "profile_url",
    ]
    display = work[[col for col in columns if col in work.columns]].copy()
    rename = {
        "staff_action": "Action",
        "data_confidence": "Confidence",
        "name": "Player",
        "position": "Pos",
        "committed_team": "Committed To",
        "model_decommit_flip_probability": "Move Risk",
        "home_city": "City",
        "home_state": "State",
        "texas_opportunity_reasons": "Why Texas",
        "data_notes": "Data Notes",
        "next_action": "Next Action",
        "profile_url": "247",
    }
    return display.rename(columns=rename)


def display_staff_ops_summary(df: pd.DataFrame) -> None:
    if df.empty or "staff_action" not in df.columns:
        return
    section_title("Action Summary")
    left, right = st.columns([1.05, 1.2])
    with left:
        summary = action_summary(df)
        if not summary.empty:
            st.dataframe(summary, width="stretch", hide_index=True)
    with right:
        queue = review_queue(df, limit=20)
        if not queue.empty:
            st.caption("Highest-priority names that need more intel or coach review.")
            st.dataframe(queue, width="stretch", hide_index=True)


def visual_board_section(df: pd.DataFrame, texas_mode: bool = False, key_prefix: str = "visual") -> None:
    section_title("Map View")
    left, right = st.columns([1.45, 0.95])
    with left:
        st.markdown('<div class="viz-title">Player Hometowns</div>', unsafe_allow_html=True)
        st.caption("Each dot is a recruit's hometown, not his committed school. Larger dots mean higher move risk. The rings show distance from Austin on the Texas board.")
        display_recruiting_map(df, texas_mode=texas_mode, key=f"{key_prefix}_home_map")
    with right:
        st.markdown('<div class="viz-title">Board Breakdown</div>', unsafe_allow_html=True)
        top_bar_chart(df, "position", "model_decommit_flip_probability_num", "Position Rooms", limit=8)
        risk_distribution_chart(df)
    chart_left, chart_right = st.columns(2)
    with chart_left:
        top_bar_chart(df, "committed_team", "model_decommit_flip_probability_num", "Committed Schools", limit=10)
    with chart_right:
        top_bar_chart(df, "home_state", "model_decommit_flip_probability_num", "Home States", limit=10)


def display_call_sheet(df: pd.DataFrame, texas_mode: bool = False) -> None:
    rows = []
    for _, row in df.iterrows():
        rank = html.escape(clean_phrase(row.get("board_rank", "")))
        name = html.escape(str(row.get("name", "")))
        position = html.escape(str(row.get("position", "")))
        stars = html.escape(str(row.get("star_bucket", "")))
        national_rank = clean_rank(row.get("national_rank", ""))
        rank_text = f"#{html.escape(national_rank)}" if national_rank != "Unranked" else "Unranked"
        player_chips = "".join(
            f'<span class="meta-chip">{chip}</span>'
            for chip in [position, stars, rank_text]
            if chip
        )
        team = html.escape(str(row.get("committed_team", "")))
        home = html.escape(home_label(row))
        committed_date = html.escape(clean_phrase(row.get("committed_date", "")))
        risk = html.escape(risk_label(row.get("model_decommit_flip_probability_num", "")))
        flip = html.escape(clean_notes(row.get("model_decommit_flip_probability", "")))
        stick = html.escape(clean_notes(row.get("model_stick_probability", "")))
        context = html.escape(short_context(row, texas_mode=texas_mode))
        full_context = html.escape(texas_angle(row) if texas_mode else short_context(row, texas_mode=False))
        activity = html.escape(activity_summary(row))
        staff = html.escape(staff_summary(row))
        quality = html.escape(data_quality_text(row, texas_mode=texas_mode))
        quality_markup = data_quality_html(row, texas_mode=texas_mode)
        risk_note = html.escape(short_risk_read(row))
        profile_risk_note = html.escape(short_risk_read(row))
        if texas_mode:
            texas_distance = html.escape(target_distance_summary(row) or "Distance not captured")
            texas_tags = html.escape(
                display_reason_list(row.get("texas_context_tags", ""), "No Texas signal captured")
            )
            owner_raw = clean_phrase(row.get("staff_owner", ""))
            next_action_raw = clean_phrase(row.get("next_action", ""))
            owner_line = f'<div class="call-muted">Owner: {html.escape(owner_raw)}</div>' if owner_raw else ""
            extra_cell = (
                f'<div class="call-player">{texas_distance}</div>'
                f'<div class="call-muted">{texas_tags}</div>'
                f'{owner_line}'
            )
            next_line = f" Next: {next_action_raw}." if next_action_raw else ""
            extra_detail = f"{texas_distance}. {texas_tags}.{html.escape(next_line)}"
        else:
            school_distance = html.escape(miles_summary(row.get("home_to_school_miles", ""), "from home") or "School distance not captured")
            distance_bucket = html.escape(clean_phrase(row.get("distance_bucket", "")) or "distance bucket not captured")
            extra_cell = (
                f'<div class="call-player">{school_distance}</div>'
                f'<div class="meta-row"><span class="meta-chip">{distance_bucket}</span></div>'
                f'<div class="call-muted">{activity}</div>'
            )
            extra_detail = full_context
        profile = html.escape(str(row.get("profile_url", "")))
        image_url = player_image_url(row.get("profile_url", ""))
        avatar = avatar_html(image_url, row.get("name", ""))
        profile_photo = avatar_html(image_url, row.get("name", ""), large=True)
        rows.append(
            f'<tr class="call-row">'
            f'<td class="call-rank">{rank}</td>'
            f'<td>'
            f'<div class="player-cell">'
            f'{avatar}'
            f'<div class="player-text"><div class="call-player">{name}</div><div class="meta-row">{player_chips}</div></div>'
            f'<div class="profile-popover">'
            f'<div class="profile-head">{profile_photo}<div>'
            f'<div class="profile-title">{name}</div>'
            f'<div class="meta-row">{player_chips}</div>'
            f'{quality_markup}'
            f'<div class="profile-line">Committed to {team}</div>'
            f'<div class="profile-line">Home: {home}</div>'
            f'<div class="profile-line">Since: {committed_date}</div>'
            f'</div></div>'
            f'<dl class="profile-grid">'
            f'<dt>Risk</dt><dd>{flip} flip / {stick} stick</dd>'
            f'<dt>Context</dt><dd>{extra_detail}</dd>'
            f'<dt>Activity</dt><dd>{activity}</dd>'
            f'<dt>Data</dt><dd>{quality}</dd>'
            f'<dt>Why</dt><dd>{profile_risk_note}</dd>'
            f'</dl>'
            f'</div>'
            f'</div>'
            f'</td>'
            f'<td><div>{team}</div><div class="commit-subline">Home: {home}</div><div class="commit-subline">Since: {committed_date}</div></td>'
            f'<td><div class="call-risk {risk_class(row)}">{risk}</div><div class="call-muted">{flip} flip</div><div class="call-muted">{risk_note}</div></td>'
            f'<td>{extra_cell}</td>'
            f'<td><a class="call-link" href="{profile}" target="_blank">247</a></td>'
            f'</tr>'
        )
    st.markdown(
        '<table class="call-sheet">'
        '<colgroup>'
        '<col style="width: 48px;">'
        '<col style="width: 31%;">'
        '<col style="width: 20%;">'
        '<col style="width: 17%;">'
        '<col style="width: 24%;">'
        '<col style="width: 52px;">'
        '</colgroup>'
        '<thead><tr>'
        '<th>#</th>'
        '<th>Player</th>'
        '<th>Commit</th>'
        '<th>Risk</th>'
        f'<th>{"Texas Context" if texas_mode else "Why Loose"}</th>'
        '<th>247</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>',
        unsafe_allow_html=True,
    )


def unique_values(*series: pd.Series) -> list[str]:
    values = []
    for item in series:
        values.extend(str(value) for value in item.dropna().unique() if str(value).strip())
    return sorted(set(values))


def sidebar_filters(target: pd.DataFrame, national: pd.DataFrame, all_commits: pd.DataFrame) -> dict[str, object]:
    distance_values = unique_values(
        target["target_distance_bucket"], national["distance_bucket"], all_commits["distance_bucket"]
    )
    confidence_values = unique_values(target["data_confidence"]) if "data_confidence" in target.columns else []

    with st.sidebar:
        st.markdown("### Filters")
        filters = {
            "board_view": st.selectbox(
                "Show",
                ["All 2027 Commitments", "Texas Targets", "National Weak Commits"],
                key="sidebar_board_view",
            ),
            "positions": st.multiselect(
                "Position",
                unique_values(target["position"], national["position"], all_commits["position"]),
                key="sidebar_position",
            ),
            "tiers": st.multiselect(
                "Flip risk",
                ["Very High", "High", "Medium", "Low"],
                default=[],
                key="sidebar_tier",
            ),
            "states": st.multiselect(
                "Home state",
                unique_values(target["home_state"], national["home_state"], all_commits["home_state"]),
                key="sidebar_state",
            ),
            "teams": st.multiselect(
                "Committed school",
                unique_values(target["committed_team"], national["committed_team"], all_commits["committed_team"]),
                key="sidebar_team",
            ),
            "confidence": st.multiselect(
                "Data confidence",
                confidence_values,
                key="sidebar_confidence",
            ),
            "distance": st.multiselect(
                "Distance from home",
                distance_values,
                key="sidebar_distance",
            ),
            "min_flip": st.slider("Minimum flip risk", 0, 100, 0, 1, key="sidebar_flip"),
            "max_rows": st.slider("Rows shown", 25, 1500, 1500, 25, key="sidebar_rows_full_board"),
            "only_activity": st.toggle(
                "Only show outside activity after commit",
                value=False,
                key="sidebar_activity",
            ),
            "staff_priority_only": st.toggle(
                "Staff priority only",
                value=False,
                key="sidebar_staff_priority",
            ),
            "hide_do_not_pursue": st.toggle(
                "Hide do-not-pursue",
                value=True,
                key="sidebar_do_not_pursue",
            ),
        }
    return filters


def filter_board(df: pd.DataFrame, filters: dict[str, object], texas_mode: bool = False) -> pd.DataFrame:
    filtered = df.copy()

    if filters["positions"]:
        filtered = filtered[filtered["position"].isin(filters["positions"])]
    if filters["tiers"]:
        filtered = filtered[
            filtered["model_decommit_flip_probability_num"].map(risk_label).isin(filters["tiers"])
        ]
    if filters["states"]:
        filtered = filtered[filtered["home_state"].isin(filters["states"])]
    if filters["teams"]:
        filtered = filtered[filtered["committed_team"].isin(filters["teams"])]
    if texas_mode and filters.get("confidence") and "data_confidence" in filtered.columns:
        filtered = filtered[filtered["data_confidence"].isin(filters["confidence"])]
    if filters["distance"]:
        distance_col = "target_distance_bucket" if texas_mode else "distance_bucket"
        filtered = filtered[filtered[distance_col].isin(filters["distance"])]
    probability_col = (
        "model_decommit_flip_probability_num"
        if "model_decommit_flip_probability_num" in filtered.columns
        else "estimated_decommit_flip_probability_num"
    )
    if probability_col in filtered.columns:
        filtered = filtered[filtered[probability_col].fillna(0) >= filters["min_flip"]]
    if filters["only_activity"]:
        activity_cols = [
            "post_commit_other_school_official_visits",
            "post_commit_other_school_visits",
            "post_commit_other_school_offers",
        ]
        existing = [col for col in activity_cols if col in filtered.columns]
        if existing:
            mask = filtered[existing].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) > 0
            filtered = filtered[mask]
    if filters["staff_priority_only"] and "staff_priority" in filtered.columns:
        filtered = filtered[filtered["staff_priority"].map(is_yes)]
    if filters["hide_do_not_pursue"] and "do_not_pursue" in filtered.columns:
        filtered = filtered[~filtered["do_not_pursue"].map(is_yes)]

    return filtered.head(int(filters["max_rows"]))


def metric_row(df: pd.DataFrame, texas_mode: bool = False) -> None:
    cols = st.columns(5)
    cols[0].metric("Players Shown", f"{len(df):,}")
    if "model_decommit_flip_probability_num" in df.columns and len(df):
        cols[1].metric("Avg Move Risk", f"{df['model_decommit_flip_probability_num'].mean():.1f}%")
    elif "estimated_decommit_flip_probability_num" in df.columns and len(df):
        cols[1].metric("Avg Move Risk", f"{df['estimated_decommit_flip_probability_num'].mean():.1f}%")
    else:
        cols[1].metric("Avg Move Risk", "n/a")
    very_high = 0
    if "model_decommit_flip_probability_num" in df.columns and len(df):
        very_high = int((df["model_decommit_flip_probability_num"].fillna(0) >= 70).sum())
    cols[2].metric("Very High Risk", f"{very_high:,}")
    activity_cols = [col for col in ["post_commit_other_school_official_visits", "post_commit_other_school_visits", "post_commit_other_school_offers"] if col in df.columns]
    activity = 0
    if activity_cols and len(df):
        activity = int((df[activity_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) > 0).sum())
    cols[3].metric("Outside Activity", f"{activity:,}")
    if "model_stick_probability_num" in df.columns and len(df):
        cols[4].metric("Avg Stay Chance", f"{df['model_stick_probability_num'].mean():.1f}%")
    else:
        cols[4].metric("Avg Stay Chance", "n/a")


def position_room_metrics(texas_df: pd.DataFrame, national_df: pd.DataFrame) -> None:
    cols = st.columns(4)
    cols[0].metric("Texas Names", f"{len(texas_df):,}")
    texas_risk = "n/a"
    if "model_decommit_flip_probability_num" in texas_df.columns and len(texas_df):
        texas_risk = f"{texas_df['model_decommit_flip_probability_num'].mean():.1f}%"
    cols[1].metric("Texas Avg Risk", texas_risk)
    cols[2].metric("National Names", f"{len(national_df):,}")
    national_risk = "n/a"
    if "model_decommit_flip_probability_num" in national_df.columns and len(national_df):
        national_risk = f"{national_df['model_decommit_flip_probability_num'].mean():.1f}%"
    cols[3].metric("National Avg Risk", national_risk)


def export_board(df: pd.DataFrame, label: str, filename: str, key: str) -> None:
    st.download_button(
        label,
        df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def model_summary_metrics(rolling: pd.DataFrame) -> None:
    clean = rolling.copy()
    for col in ["auc", "top_20_hit_rate", "actual_decommit_flip_rate", "average_predicted_probability"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    holdout = clean[clean["test_year"].astype(str).isin(["2025", "2026"])]
    view = holdout if not holdout.empty else clean

    cols = st.columns(4)
    cols[0].metric("Clean Test Classes", f"{len(view):,}")
    cols[1].metric("Top 20 Who Broke Commitment", f"{view['top_20_hit_rate'].mean() * 100:.1f}%")
    cols[2].metric("All Commits Who Broke Commitment", f"{view['actual_decommit_flip_rate'].mean() * 100:.1f}%")
    cols[3].metric("Board's Avg Risk", f"{view['average_predicted_probability'].mean() * 100:.1f}%")


def model_scorecard(rolling: pd.DataFrame, clean_tests_only: bool = True) -> pd.DataFrame:
    """Translate validation output into a staff-readable past-class scorecard."""
    clean = rolling.copy()
    if clean_tests_only:
        clean = clean[clean["test_year"].astype(str).isin(["2025", "2026"])]
    for col in ["top_20_hit_rate", "actual_decommit_flip_rate", "average_predicted_probability"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    columns = pd.DataFrame(
        {
            "Test Class": clean["test_year"].astype(str),
            "Top 20 Who Broke Commitment": clean["top_20_hit_rate"].map(lambda value: f"{value:.1%}"),
            "All Commits Who Broke Commitment": clean["actual_decommit_flip_rate"].map(lambda value: f"{value:.1%}"),
            "Board's Avg Risk": clean["average_predicted_probability"].map(lambda value: f"{value:.1%}"),
        }
    )
    return columns


def load_commit_calendar() -> pd.DataFrame:
    """Return avg commits per calendar day (MM-DD) across completed classes."""
    path = DATA_DIR / "summary_by_day.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df["class_year"] = pd.to_numeric(df["class_year"], errors="coerce")
    df["commits"] = pd.to_numeric(df["commits"], errors="coerce").fillna(0)
    df = df[df["class_year"].isin([2022, 2023, 2024, 2025])].copy()
    df["dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["dt"])
    df["prime_year"] = df["class_year"] - 1
    df = df[df["dt"].dt.year == df["prime_year"]]
    df["mmdd"] = df["dt"].dt.strftime("%m-%d")
    df["month"] = df["dt"].dt.month
    df = df[df["month"].isin([5, 6, 7, 8])]
    summary = (
        df.groupby("mmdd", as_index=False)["commits"]
        .sum()
        .assign(avg=lambda x: x["commits"] / 4)
    )
    summary["month"] = summary["mmdd"].str[:2].astype(int)
    summary["day"] = summary["mmdd"].str[3:].astype(int)
    summary = summary.sort_values(["month", "day"]).reset_index(drop=True)
    return summary


def crowd_zone(avg: float) -> tuple[str, str]:
    if avg >= 30:
        return "Peak Crowd", "#8F3F00"
    if avg >= 20:
        return "Crowded", "#BF5700"
    if avg >= 10:
        return "Moderate", "#B08040"
    return "Stand Out", "#2F6B4F"


def commit_timing_tab(calendar: pd.DataFrame) -> None:
    section_title("Commitment Announcement Calendar")
    st.markdown(
        '<div class="briefing-box"><strong>What this is</strong><br>'
        "How many ranked high school recruits announce their commitment on each day of the summer, "
        "averaged across the 2022–2025 signing classes. Use this to show recruits exactly how "
        "crowded their target date is — and where the quiet windows are.</div>",
        unsafe_allow_html=True,
    )

    if calendar.empty:
        st.warning("summary_by_day.csv not found. Run scrape_247_commit_dates.py first.")
        return

    # ── Key facts row ────────────────────────────────────────────────────────
    june_avg = calendar[calendar["month"] == 6]["avg"].sum()
    july_avg = calendar[calendar["month"] == 7]["avg"].sum()
    late_june_avg = calendar[
        (calendar["month"] == 6) & (calendar["day"] >= 21) & (calendar["day"] <= 28)
    ]["avg"].mean()
    july4_avg = calendar[(calendar["month"] == 7) & (calendar["day"] == 4)]["avg"].values
    july4_val = float(july4_avg[0]) if len(july4_avg) else 0.0
    aug1_avg = calendar[(calendar["month"] == 8) & (calendar["day"] == 1)]["avg"].values
    aug1_val = float(aug1_avg[0]) if len(aug1_avg) else 0.0

    early_june_avg = float(
        calendar[(calendar["month"] == 6) & (calendar["day"] <= 15)]["avg"].mean()
    )

    cols = st.columns(5)
    cols[0].metric("June commits / year", f"{june_avg:.0f}")
    cols[1].metric("July commits / year", f"{july_avg:.0f}", delta=f"{july_avg - june_avg:.0f} vs June", delta_color="off")
    cols[2].metric("Late June peak (Jun 21-28)", f"{late_june_avg:.0f}/day")
    cols[3].metric("July 4 commits / year", f"{july4_val:.0f}")
    cols[4].metric("Aug 1 commits / year", f"{aug1_val:.0f}")

    st.markdown(
        '<div class="briefing-box">'
        "<strong>The corrected picture</strong><br>"
        f"June is <em>nearly 2× more crowded than July</em> overall ({june_avg:.0f} vs {july_avg:.0f} commits/year). "
        f"Late June (Jun 21–28) is the busiest window — an average of {late_june_avg:.0f} ranked recruits announce each of those days nationally. "
        f"Early June (Jun 1–15) averages only {early_june_avg:.0f}/day — genuinely quiet. "
        f"July 4 sees ~{july4_val:.0f} announcements nationally and is comparable to a late-June day, not a special moment. "
        f"August 1 averages ~{aug1_val:.0f}. "
        "The real quiet windows for standing out are <strong>early June (Jun 1–15)</strong> and <strong>mid-to-late July</strong>."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Summer calendar bar chart ────────────────────────────────────────────
    section_title("Day-by-Day Summer Calendar")
    month_filter = st.radio(
        "Show month",
        ["All Summer", "May", "June", "July", "August"],
        horizontal=True,
        key="timing_month_filter",
    )
    month_map = {"May": 5, "June": 6, "July": 7, "August": 8}
    if month_filter == "All Summer":
        chart_df = calendar.copy()
    else:
        chart_df = calendar[calendar["month"] == month_map[month_filter]].copy()

    chart_df["label"] = chart_df.apply(
        lambda r: f"{pd.Timestamp(2000, int(r['month']), int(r['day'])).strftime('%b %-d')}",
        axis=1,
    )
    chart_df["zone"] = chart_df["avg"].map(lambda v: crowd_zone(v)[0])
    chart_df["color"] = chart_df["avg"].map(lambda v: crowd_zone(v)[1])

    highlight_dates = {"07-04", "08-01", "06-21", "06-22", "06-23", "06-24", "06-25", "06-26", "06-27", "06-28"}
    chart_df["highlight"] = chart_df["mmdd"].isin(highlight_dates)

    bar = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=None, title="", axis=alt.Axis(labelAngle=-60, labelFontSize=10)),
            y=alt.Y("avg:Q", title="Avg commits / year"),
            color=alt.Color(
                "zone:N",
                scale=alt.Scale(
                    domain=["Stand Out", "Moderate", "Crowded", "Peak Crowd"],
                    range=["#2F6B4F", "#B08040", "#BF5700", "#8F3F00"],
                ),
                legend=alt.Legend(title="Crowd Level"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Date"),
                alt.Tooltip("avg:Q", title="Avg commits/year", format=".1f"),
                alt.Tooltip("zone:N", title="Crowd level"),
            ],
        )
        .properties(height=320, title="Commitment announcements per day — averaged across 2022–2025 classes")
    )
    st.altair_chart(bar, use_container_width=True)

    st.caption(
        "Data: 247Sports composite high school rankings, 2022–2025 signing classes. "
        "Each bar = average number of ranked recruits who announced that calendar day across those 4 classes."
    )

    # ── Date lookup tool ─────────────────────────────────────────────────────
    section_title("Date Lookup — Recruit Pitch Tool")
    st.markdown(
        '<div class="briefing-box"><strong>How to use this on a visit</strong><br>'
        "Enter a recruit's target announcement date below. The tool will show exactly how crowded that day is "
        "historically and generate a talking-point summary you can use in the room.</div>",
        unsafe_allow_html=True,
    )

    col_month, col_day = st.columns(2)
    with col_month:
        lookup_month = st.selectbox(
            "Month",
            [5, 6, 7, 8],
            index=1,
            format_func=lambda m: {5: "May", 6: "June", 7: "July", 8: "August"}[m],
            key="timing_lookup_month",
        )
    with col_day:
        import calendar as cal_mod
        max_day = cal_mod.monthrange(2024, lookup_month)[1]
        lookup_day = st.number_input("Day", min_value=1, max_value=max_day, value=4 if lookup_month == 7 else 15, key="timing_lookup_day")

    mmdd = f"{lookup_month:02d}-{int(lookup_day):02d}"
    match = calendar[calendar["mmdd"] == mmdd]
    lookup_avg = float(match["avg"].values[0]) if not match.empty else 0.0
    zone_label, zone_color = crowd_zone(lookup_avg)

    date_str = f"{pd.Timestamp(2024, lookup_month, int(lookup_day)).strftime('%B %-d')}"

    st.markdown(
        f'<div style="background:#FFFFFF;border:1px solid #D8D8D8;border-left:6px solid {zone_color};'
        f'padding:18px 20px;margin:12px 0;">'
        f'<div style="font-size:0.78rem;font-weight:900;text-transform:uppercase;color:{zone_color};'
        f'letter-spacing:0.1em;margin-bottom:6px;">{zone_label}</div>'
        f'<div style="font-size:2rem;font-weight:950;color:#111;">{lookup_avg:.0f} recruits</div>'
        f'<div style="color:#555;font-size:0.9rem;margin-top:4px;">announce nationally on {date_str} in an average year</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Talking points
    early_june_avg_lookup = float(
        calendar[(calendar["month"] == 6) & (calendar["day"] <= 15)]["avg"].mean()
    )
    if lookup_avg >= 30:
        pitch = (
            f"**{date_str} is the peak of the commitment calendar.** "
            f"An average of {lookup_avg:.0f} ranked recruits nationally announce on this date each year — "
            f"that's {lookup_avg / max(early_june_avg_lookup, 1):.1f}× more than a quiet early-June day ({early_june_avg_lookup:.0f}/day). "
            f"Every recruiting outlet is covering a pile of announcements at once. "
            f"Committing publicly earlier — when early June sees only ~{early_june_avg_lookup:.0f} kids announcing nationally — "
            f"means the announcement gets real oxygen instead of getting buried."
        )
    elif lookup_avg >= 20:
        pitch = (
            f"**{date_str} is a crowded announcement day.** "
            f"An average of {lookup_avg:.0f} ranked recruits announce on this date nationally each year. "
            f"That's {lookup_avg / max(early_june_avg_lookup, 1):.1f}× more competition for coverage than early June (~{early_june_avg_lookup:.0f}/day). "
            f"Committing on-site now means the announcement has far less competition for attention."
        )
    elif lookup_avg >= 10:
        pitch = (
            f"**{date_str} is a moderate day — about {lookup_avg:.0f} ranked recruits nationally.** "
            f"It's not the noisiest time, but the late-June wave peaks at ~{late_june_avg:.0f}/day. "
            f"Early June right now is only ~{early_june_avg_lookup:.0f}/day — "
            f"a public commitment now still gets more individual attention than waiting."
        )
    else:
        pitch = (
            f"**{date_str} is actually a quiet day — only about {lookup_avg:.0f} ranked recruits announce nationally.** "
            f"If a recruit is holding out for this date to \"make a splash,\" the data supports that it's genuinely low-noise. "
            f"The counter-pitch: right now (early June) is also ~{early_june_avg_lookup:.0f}/day — "
            f"equally quiet, and gets the public commitment done weeks sooner."
        )

    st.markdown(pitch)

    # ── Monthly summary table ─────────────────────────────────────────────────
    with st.expander("Monthly summary (completed classes 2022–2025)"):
        month_summary = (
            calendar.groupby("month", as_index=False)
            .agg(total_avg=("avg", "sum"), peak_day_avg=("avg", "max"))
            .assign(month_name=lambda d: d["month"].map({5: "May", 6: "June", 7: "July", 8: "August"}))
            .assign(total_avg=lambda d: d["total_avg"].map(lambda v: f"{v:.0f}"))
            .assign(peak_day_avg=lambda d: d["peak_day_avg"].map(lambda v: f"{v:.0f}/day"))
        )
        peak_days = (
            calendar.sort_values("avg", ascending=False)
            .groupby("month", as_index=False)
            .first()[["month", "mmdd", "avg"]]
        )
        peak_days["peak_date"] = peak_days["mmdd"].map(
            lambda s: pd.Timestamp(2024, int(s[:2]), int(s[3:])).strftime("%b %-d")
        )
        peak_days["peak_avg"] = peak_days["avg"].map(lambda v: f"{v:.0f}/day")
        merged = month_summary.merge(
            peak_days[["month", "peak_date", "peak_avg"]], on="month", how="left"
        ).rename(
            columns={
                "month_name": "Month",
                "total_avg": "Total Commits / Year",
                "peak_day_avg": "Busiest Single Day",
                "peak_date": "Busiest Date",
                "peak_avg": "Busiest Day Volume",
            }
        )[["Month", "Total Commits / Year", "Busiest Date", "Busiest Day Volume"]]
        st.dataframe(merged, hide_index=True, use_container_width=True)

    # ── Year-over-year trend ─────────────────────────────────────────────────
    with st.expander("Year-over-year trend — is June getting more crowded?"):
        raw = pd.read_csv(DATA_DIR / "summary_by_day.csv", low_memory=False)
        raw["class_year"] = pd.to_numeric(raw["class_year"], errors="coerce")
        raw["commits"] = pd.to_numeric(raw["commits"], errors="coerce").fillna(0)
        raw = raw[raw["class_year"].isin([2022, 2023, 2024, 2025])].copy()
        raw["dt"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["dt"])
        raw["prime_year"] = raw["class_year"] - 1
        raw = raw[raw["dt"].dt.year == raw["prime_year"]]
        raw["month"] = raw["dt"].dt.month
        yoy = raw[raw["month"].isin([6, 7])].copy()
        yoy["Month"] = yoy["month"].map({6: "June", 7: "July"})
        yoy_summary = (
            yoy.groupby(["class_year", "Month"], as_index=False)["commits"]
            .sum()
            .rename(columns={"class_year": "Class", "commits": "Commits"})
        )
        yoy_chart = (
            alt.Chart(yoy_summary)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("Class:O", title="Signing Class"),
                y=alt.Y("Commits:Q", title="Commits in month"),
                color=alt.Color(
                    "Month:N",
                    scale=alt.Scale(domain=["June", "July"], range=["#BF5700", "#555555"]),
                ),
                tooltip=["Class:O", "Month:N", "Commits:Q"],
            )
            .properties(height=260, title="June vs July commit volume by class year")
        )
        st.altair_chart(yoy_chart, use_container_width=True)
        st.caption(
            "June is growing faster than July. Each class brings more early-summer commitments, "
            "making the late-June window even more crowded year over year."
        )


def legacy_main() -> None:
    inject_theme()
    target = prepare_board(load_csv(str(BOARD_DIR / "target_board_texas_2027.csv")))
    national = prepare_board(load_csv(str(BOARD_DIR / "boss_view_2027_power_team_commits.csv")))
    model_validation = load_csv(str(DATA_DIR / "flip_model_validation.csv"))
    model_buckets = load_csv(str(DATA_DIR / "flip_model_probability_buckets.csv"))
    model_events = load_csv(str(DATA_DIR / "flip_model_events.csv"))
    rolling_backtest = load_csv(str(DATA_DIR / "flip_model_rolling_backtest.csv"))
    timeline_events = load_csv(str(DATA_DIR / "timeline_events.csv"))
    commit_calendar = load_commit_calendar()
    hero()
    filters = sidebar_filters(target, national)

    tab_target, tab_national, tab_positions, tab_validation, tab_timing = st.tabs(
        ["Texas Targets", "National Weak Commits", "Position Rooms", "Model Check", "Commit Timing"]
    )

    with tab_target:
        section_title("Texas Target Board")
        briefing_box(
            "Who should Texas spend time on?",
            "This board puts Texas-relevant players first. A recruit can be loose nationally and still sit lower here if there is not a clear Texas reason to chase him.",
        )
        filtered = filter_board(target, filters, texas_mode=True)
        metric_row(filtered, texas_mode=True)
        st.caption("Sorted by Texas action tier first, then move risk.")
        export_board(filtered, "Export filtered Texas board", "texas_opportunity_board.csv", "export_texas")
        weekly_path = BOARD_DIR / "weekly_brief_texas_2027.csv"
        if weekly_path.exists():
            weekly = load_csv(str(weekly_path))
            st.download_button(
                "Download weekly staff brief",
                weekly.to_csv(index=False).encode("utf-8"),
                file_name="weekly_texas_staff_brief.csv",
                mime="text/csv",
                key="export_weekly_texas",
            )
        visual_board_section(filtered, texas_mode=True, key_prefix="texas")
        section_title("Recruiting Timeline")
        display_player_timeline(filtered, timeline_events, key="texas_player_timeline")
        section_title("Player List")
        display_call_sheet(filtered, texas_mode=True)

    with tab_national:
        section_title("National Weak Commits")
        briefing_box(
            "Where could the market move?",
            "This board shows weak commitments nationally. It is a scouting list, not a Texas target list.",
        )
        filtered = filter_board(national, filters, texas_mode=False)
        metric_row(filtered)
        st.caption("National vulnerability ranking. These players may be loose nationally, but they are not automatically Texas targets.")
        export_board(filtered, "Export filtered national board", "national_flip_risk_board.csv", "export_national")
        visual_board_section(filtered, texas_mode=False, key_prefix="national")
        section_title("Recruiting Timeline")
        display_player_timeline(filtered, timeline_events, key="national_player_timeline")
        section_title("Player List")
        display_call_sheet(filtered, texas_mode=False)

    with tab_positions:
        section_title("Position Rooms")
        briefing_box(
            "Give each position coach a clean list.",
            "Pick a room to see Texas action names and national weak commits for that position only. The left sidebar filters still apply, except this room picker controls position.",
        )
        positions = unique_values(target["position"], national["position"])
        default_position = filters["positions"][0] if filters["positions"] else (positions[0] if positions else "")
        default_index = positions.index(default_position) if default_position in positions else 0
        selected_position = st.selectbox("Position room", positions, index=default_index)
        room_filters = dict(filters)
        room_filters["positions"] = [selected_position]
        texas_room = filter_board(target, room_filters, texas_mode=True)
        national_room = filter_board(national, room_filters, texas_mode=False)
        position_room_metrics(texas_room, national_room)

        export_left, export_right = st.columns(2)
        with export_left:
            export_board(
                texas_room,
                f"Export Texas {selected_position} room",
                f"texas_{selected_position.lower()}_room.csv",
                f"export_texas_room_{selected_position}",
            )
        with export_right:
            export_board(
                national_room,
                f"Export national {selected_position} room",
                f"national_{selected_position.lower()}_room.csv",
                f"export_national_room_{selected_position}",
            )

        section_title(f"Texas {selected_position} Board")
        st.caption("Texas action names for this position room.")
        visual_board_section(texas_room, texas_mode=True, key_prefix=f"texas_room_{selected_position}")
        display_call_sheet(texas_room, texas_mode=True)

        section_title(f"National {selected_position} Weak Commits")
        st.caption("Nationally loose commits at this position. Not automatically Texas targets.")
        visual_board_section(national_room, texas_mode=False, key_prefix=f"national_room_{selected_position}")
        display_call_sheet(national_room, texas_mode=False)

    with tab_validation:
        section_title("Model Check")
        briefing_box(
            "Can staff trust the board?",
            "The cleanest check is simple: train on old classes, test the next class, then see how many of the top-ranked weak commits actually moved.",
        )

        section_title("Past Class Test")
        model_summary_metrics(rolling_backtest)
        rolling_chart = rolling_chart_data(rolling_backtest)
        roll_left, roll_right = st.columns(2)
        with roll_left:
            st.caption("Actual movement compared with the model's predicted risk")
            st.bar_chart(
                rolling_chart.set_index("Test Class")[["Actual Flip Rate", "Avg Board Risk"]],
                height=280,
            )
        with roll_right:
            st.caption("How well the board ranked weak commits near the top")
            st.bar_chart(
                rolling_chart.set_index("Test Class")[["Separated Movers From Stayers", "Top 20 That Moved"]],
                height=280,
            )
        with st.expander("Past class test numbers"):
            st.dataframe(rolling_backtest, width="stretch", hide_index=True)

        section_title("Where The Board Misses")
        review_years = [
            str(year)
            for year in sorted(
                pd.to_numeric(model_events["class_year"], errors="coerce").dropna().astype(int).unique()
            )
            if year <= 2026
        ]
        review_default_year = review_years.index("2025") if "2025" in review_years else 0
        review_left, review_right = st.columns(2)
        with review_left:
            selected_review_year = st.selectbox("Review class", review_years, index=review_default_year)
        with review_right:
            selected_review_type = st.selectbox(
                "Review type",
                [
                    "Ended elsewhere but missed",
                    "Temporary decommit/recommit",
                    "Flagged but stayed",
                    "Correctly flagged",
                    "Correctly stable",
                ],
            )
        if int(selected_review_year) <= 2024:
            st.warning("This class helped train the model, so use it as a learning example rather than clean proof.")
        else:
            st.caption("This class was not used for training, so misses here are the fairest test.")
        review_examples = model_review_examples(
            model_events,
            timeline_events,
            int(selected_review_year),
            selected_review_type,
            limit=50,
        )
        st.dataframe(review_examples, width="stretch", hide_index=True)

        section_title("Player Examples")
        example_years = [
            str(year)
            for year in sorted(
                pd.to_numeric(model_events["class_year"], errors="coerce").dropna().astype(int).unique()
            )
            if year <= 2026
        ]
        default_year_index = example_years.index("2025") if "2025" in example_years else 0
        selected_example_year = st.selectbox("Example class", example_years, index=default_year_index)
        if int(selected_example_year) <= 2024:
            st.warning(
                "This class was used to train the model. Treat these as examples of what the model learned, not clean validation."
            )
        else:
            st.caption("This class was not used to train the model, so these are cleaner validation examples.")
        examples = historical_player_examples(
            model_events,
            timeline_events,
            int(selected_example_year),
            limit=50,
        )
        st.dataframe(examples, width="stretch", hide_index=True)

        with st.expander("Technical validation tables"):
            history = model_year_history(model_events)
            st.markdown("**Year-by-year model read**")
            st.dataframe(history, width="stretch", hide_index=True)
            st.markdown("**Train / holdout split**")
            st.dataframe(model_validation, width="stretch", hide_index=True)
            st.markdown("**Overall probability buckets**")
            st.dataframe(model_buckets, width="stretch", hide_index=True)
            st.markdown("**Past classes by risk bucket**")
            st.dataframe(
                model_year_bucket_history(model_events),
                width="stretch",
                hide_index=True,
            )


    with tab_timing:
        commit_timing_tab(commit_calendar)

def main() -> None:
    """Render one recruiting board; supporting research stays out of the way."""
    inject_theme()
    target = prepare_board(load_csv(str(BOARD_DIR / "target_board_texas_2027.csv")))
    national = prepare_board(load_csv(str(BOARD_DIR / "boss_view_2027_power_team_commits.csv")))
    all_commits = prepare_board(load_csv(str(BOARD_DIR / "flip_board_all_commits.csv")))
    all_commits = all_commits[all_commits["class_year"].astype(str) == "2027"].copy()
    rolling_backtest = load_csv(str(DATA_DIR / "flip_model_rolling_backtest.csv"))
    timeline_events = load_csv(str(DATA_DIR / "timeline_events.csv"))

    hero()
    filters = sidebar_filters(target, national, all_commits)
    texas_mode = filters["board_view"] == "Texas Targets"
    board = (
        target
        if texas_mode
        else national
        if filters["board_view"] == "National Weak Commits"
        else all_commits
    )
    filtered = filter_board(board, filters, texas_mode=texas_mode)

    board_title = filters["board_view"]
    section_title(board_title)
    briefing_box(
        "What am I looking at?",
        "All 2027 Commitments is the complete board. Texas Targets puts Texas-relevant recruits first. "
        "National Weak Commits is the P4 market. Use the sidebar to narrow by position, school, geography, "
        "outside activity, or move risk.",
    )
    metric_row(filtered, texas_mode=texas_mode)
    st.caption("Sorted by move risk. Position is a filter, so there is no separate position-room page.")
    export_board(
        filtered,
        "Download this list",
        "texas_targets.csv"
        if texas_mode
        else "national_weak_commits.csv"
        if filters["board_view"] == "National Weak Commits"
        else "all_2027_commitments.csv",
        "export_current_board",
    )

    tab_list, tab_player, tab_map, tab_model, tab_timing = st.tabs(
        ["Player List", "Recruiting History", "Map", "Model Check", "Commit Timing"]
    )

    with tab_list:
        display_call_sheet(filtered, texas_mode=texas_mode)

    with tab_player:
        section_title("Recruiting History")
        display_player_timeline(filtered, timeline_events, key="current_board_player_timeline")

    with tab_map:
        visual_board_section(filtered, texas_mode=texas_mode, key_prefix="current_board")

    with tab_model:
        section_title("How This Board Has Done")
        briefing_box(
            "Did it identify weak commitments?",
            "We taught the board using older recruiting classes, then tested it on the 2025 and 2026 classes it had not seen. A broken commitment means the player later decommitted or changed schools.",
        )
        model_summary_metrics(rolling_backtest)
        st.caption("The key number: among the 20 names the board ranked loosest, how many later broke their original commitment.")
        st.dataframe(model_scorecard(rolling_backtest), width="stretch", hide_index=True)
        with st.expander("Older training-class research"):
            st.caption("These older classes helped teach the model, so they are examples, not the clean test shown above.")
            st.dataframe(rolling_backtest, width="stretch", hide_index=True)

    with tab_timing:
        commit_timing_tab(load_commit_calendar())


if __name__ == "__main__":
    main()
