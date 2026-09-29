#!/usr/bin/env python3
"""Train a "will this commitment break in the next 60 days?" model on weekly-board snapshots.

train_flip_model.py scores each commitment once, on the day it was made, so a recruit who commits and
then takes official visits elsewhere looks exactly like one who shut his recruitment down. This model
instead takes a snapshot of every active commitment on a two-week calendar grid (what a staff flip
board would show that day) and uses everything known up to that date — including visits and offers
since committing — to predict a decommit or flip within the next HORIZON_DAYS.

Leakage rules:
- A snapshot only sees timeline events dated on or before the snapshot date (247 lists scheduled
  official visits ahead of time; a future-dated visit is not counted until its date).
- Class crowding (other commits at the same school and position) is counted as of the snapshot date
  from the timeline, not from final classes.
- A snapshot is only labeled when its whole window has been observed (snapshot + HORIZON_DAYS <=
  the data cutoff), so recent commitments that haven't had time to break don't count as "stuck".

Outputs (data/):
- flip_snapshot_validation.csv       AUC / Brier / log loss by sample, vs the at-commit model
- flip_snapshot_board_backtest.csv   per test class: top-25 board hit rate on each grid date, averaged
- flip_snapshot_calibration.csv      predicted vs actual 60-day break rate by bucket (test classes)
- flip_boards/live_flip_risk.csv     every active commitment scored as of the data cutoff, with reasons
- flip_snapshot_summary.md           the above in words
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from backtest_commitment_strength import classify_outcome, events_by_player
from build_flip_boards import POWER_TEAMS
from geography_features import geography_features
from score_commitment_strength import (
    dedupe_commits,
    early_signing_day,
    high_school_cycle_cutoff,
    in_high_school_cycle,
    load_csv,
    norm_team,
    parse_date,
    rating_float,
)

HORIZON_DAYS = 60
GRID_DAYS = 14
BOARD_SIZE = 25
TEST_YEARS = [2025, 2026, 2027]  # each tested on a model trained on every earlier class
NO_VISIT_DAYS = 999
SIGNING_PERIOD_DAYS = 60

NUMERIC_FEATURES = [
    # Known on commitment day (the at-commit model's inputs, crowding now as of the snapshot).
    "rating",
    "prior_commitments_at_commit",
    "prior_decommits_at_commit",
    "offers_before_commit",
    "official_visits_before_commit",
    "unofficial_visits_before_commit",
    "home_to_school_miles",
    # Timing at the snapshot.
    "days_committed",
    "days_to_signing",
    "commit_days_to_signing",
    # Since committing, up to the snapshot.
    "official_visits_elsewhere",
    "official_visits_elsewhere_power",
    "unofficial_visits_elsewhere",
    "official_visits_committed_school",
    "unofficial_visits_committed_school",
    "offers_since_commit",
    "power_offers_since_commit",
    "days_since_last_visit_elsewhere",
    "visits_elsewhere_last_30_days",
    "other_schools_visited",
    # Class context at the snapshot.
    "same_position_commits_now",
    "higher_rated_same_position_now",
    "class_commits_now",
    # The program: others decommitting lately (turmoil, a coaching change) and how he compares with
    # its usual recruit (a 4-star at a program that signs 3-stars is a flip target).
    "school_decommits_last_60_days",
    "school_class_decommits_last_60_days",
    "program_rating_level",
    "rating_above_program",
    # Geography of the schools he's visited since committing, against the one he's committed to: a New
    # Orleans kid committed to Texas (459 miles) who keeps visiting LSU (75 miles) has a closer option.
    "nearest_visited_elsewhere_miles",
    "visited_school_miles_closer",
    "visited_home_state_school",
    # His school's head coach fired or gone (data/coaching_changes.csv, from fetch_coaching_changes.py).
    "coach_change_since_commit",
    "days_since_coach_change",
    # Insider coverage (extract_insider_signals.py over the backfilled On3/Rivals coverage and the web reports).
    # Unknown (not zero) before coverage starts, so "no talk" in years with no coverage isn't read as "quiet".
    "insider_articles_30d",
    "insider_flip_talk_30d",
    "insider_flip_talk_90d",
    "insider_firm_talk_30d",
    "insider_other_school_flip_30d",
    "insider_other_school_prediction_90d",
    # Who's saying it: his own school's beat worrying about a flip is a different signal from a rival
    # school's site hoping for one, or national coverage.
    "insider_own_site_flip_90d",
    "insider_national_flip_30d",
    "insider_rival_site_flip_30d",
    # A decision coming soon, and whether it comes with flip talk.
    "insider_decision_talk_30d",
    "insider_decision_flip_30d",
    # Momentum: coverage in the last two weeks against the pace of the two months before.
    "insider_articles_14d",
    "insider_momentum",
    # On3/Rivals insiders' dated picks (backfill_predictions.py): as of the snapshot date only. Unknown for
    # players outside On3's ranked lists; learned from outcomes like everything else, not copied.
    "on3_picks_other_school",
    "on3_switched_away",
    "on3_best_accuracy_other",
    "on3_max_confidence_other",
]
CATEGORICAL_FEATURES = ["position", "star_bucket", "distance_bucket", "is_in_state_commit", "committed_power_team"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def grid_dates(start: date, end: date) -> list[date]:
    """Every GRID_DAYS-th Monday from an anchor, between start and end inclusive."""
    anchor = date(2019, 1, 7)  # a Monday
    first = anchor + timedelta(days=((start - anchor).days + GRID_DAYS - 1) // GRID_DAYS * GRID_DAYS)
    out = []
    day = first
    while day <= end:
        out.append(day)
        day += timedelta(days=GRID_DAYS)
    return out


def build_commitments(recruits: dict, grouped: dict, cutoff: date) -> list[dict]:
    """Every commitment event with its end (decommit/flip date, or when it stopped being at risk)."""
    commitments = []
    for key, player_events in grouped.items():
        recruit = recruits.get(key)
        if not recruit:
            continue
        class_year = int(key[0])
        # Two commitments on the same day (Shumaker: Colorado, then back to Ole Miss the same day) can't be
        # ordered by date; keep the one matching his current commitment and drop the other as ambiguous.
        by_day: dict[str, list[dict]] = {}
        for event in player_events:
            if event["event_type"] == "Commitment":
                by_day.setdefault(event["event_date"], []).append(event)
        current = norm_team(recruit.get("committed_team", ""))
        ambiguous = {id(e) for same in by_day.values() if len({norm_team(x["school"]) for x in same}) > 1
                     for e in same if not (current and (norm_team(e["school"]) == current or norm_team(e["school"]).startswith(current + " ")))}
        for event in player_events:
            if event["event_type"] != "Commitment" or id(event) in ambiguous:
                continue
            start = parse_date(event["event_date"])
            if not start:
                continue
            outcome, outcome_date, _ = classify_outcome(event, player_events)
            broke_on = parse_date(outcome_date) if outcome in {"decommitted", "flipped"} else None
            # At risk until it breaks, signs, or the signing period ends. Breaks cluster in the month
            # before early signing day and almost never come more than ~60 days after it; many older
            # commitments have no signing recorded, and counting them "at risk" through the summer
            # would pad the older classes with months of empty weeks.
            settled_on = parse_date(outcome_date) if outcome == "stuck" else None
            signing_period_end = early_signing_day(class_year) + timedelta(days=SIGNING_PERIOD_DAYS)
            end = min(d for d in [broke_on, settled_on, signing_period_end, high_school_cycle_cutoff(class_year)] if d)
            commitments.append({
                "key": key, "class_year": class_year, "recruit": recruit, "event": event,
                "school": event["school"], "school_norm": norm_team(event["school"]),
                "start": start, "end": end, "broke_on": broke_on, "outcome": outcome,
                "events": player_events,
            })
    return commitments


def crowding_index(commitments: list[dict]) -> dict:
    """(class, school, position) → commit intervals, to count who else is committed on a date."""
    index: dict[tuple, list] = defaultdict(list)
    for c in commitments:
        index[(c["class_year"], c["school_norm"], c["recruit"].get("position", ""))].append(
            (c["start"], c["broke_on"] or date.max, rating_float(c["recruit"]), c["key"][1]))
    return index


def class_index(commitments: list[dict]) -> dict:
    index: dict[tuple, list] = defaultdict(list)
    for c in commitments:
        index[(c["class_year"], c["school_norm"])].append((c["start"], c["broke_on"] or date.max, c["key"][1]))
    return index


def program_levels(commitments: list[dict]) -> dict:
    """(school, class) → average rating of the school's commitments in the classes before it."""
    by_school: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for c in commitments:
        rating = rating_float(c["recruit"])
        if rating:
            by_school[c["school_norm"]][c["class_year"]].append(rating)
    levels = {}
    for school, years in by_school.items():
        for year in range(min(years), max(years) + 2):
            earlier = [r for y, ratings in years.items() if y < year for r in ratings]
            if len(earlier) >= 5:
                levels[(school, year)] = sum(earlier) / len(earlier)
    return levels


def school_breaks(commitments: list[dict]) -> dict:
    """school → sorted (break date, class, player) for every decommit/flip from it."""
    index: dict[str, list] = defaultdict(list)
    for c in commitments:
        if c["broke_on"]:
            index[c["school_norm"]].append((c["broke_on"], c["class_year"], c["key"][1]))
    for school in index:
        index[school].sort()
    return index


COACHING: dict[str, list[date]] = {}   # school key → dates its head coach was fired or left
SCHOOL_KEYS: dict[str, str] = {}       # norm_team(timeline school name) → coaching-change school key


def load_coaching_changes(path: Path, commit_dates: Path, timeline: list[dict]) -> int:
    from fetch_coaching_changes import school_key, timeline_school_keys
    if not path.exists():
        return 0
    SCHOOL_KEYS.update(timeline_school_keys(commit_dates, timeline))
    for row in load_csv(path):
        if row["date"]:
            COACHING.setdefault(row["school_key"], []).append(date.fromisoformat(row["date"]))
    for key in COACHING:
        COACHING[key].sort()
    return sum(len(v) for v in COACHING.values())


INSIDER: dict[str, list[tuple]] = {}   # player id → [(date, flip, firm, prediction, schools, source school, decision, prediction schools)]
INSIDER_COVERAGE_START = date(2022, 8, 1)   # backfilled On3/Rivals and 247 coverage begins
ON3_PICKS: dict[str, list[dict]] = {}       # 247 player id → On3 insider picks


def load_insider_signals(path: Path) -> int:
    if not path.exists():
        return 0
    for row in load_csv(path):
        INSIDER.setdefault(row["player_id"], []).append((date.fromisoformat(row["date"]), int(row["flip_talk"]), int(row["firm_talk"]),
                                                          int(row["prediction_talk"]), tuple(s for s in row["schools"].split("|") if s),
                                                          row.get("source_school", "unknown"), int(row.get("decision_talk") or 0),
                                                          tuple(s for s in (row.get("prediction_schools") or "").split("|") if s)))
    for rows in INSIDER.values():
        rows.sort()
    return sum(len(v) for v in INSIDER.values())


def load_on3_picks(folder: Path) -> int:
    """On3 insider picks keyed by 247 player id (name + class match), schools in the model's naming."""
    import json
    from extract_insider_signals import school_patterns, schools_in
    if not folder.exists():
        return 0
    patterns = school_patterns()
    to_school = lambda name: (schools_in(name, patterns) or [norm_team(name)])[0] if name else ""
    ids: dict[tuple[int, str], list[str]] = {}
    for r in load_csv(Path("data/raw_rankings.csv")):
        ids.setdefault((int(r["class_year"]), " ".join(r["name"].lower().replace(".", "").split())), []).append(r["player_id"])
    for path in sorted(folder.glob("on3_picks-*.jsonl")):
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            match = ids.get((row["class_year"], " ".join(row["name"].lower().replace(".", "").split())), [])
            if len(match) != 1:
                continue
            ON3_PICKS[match[0]] = [{**pick, "school": to_school(pick["school"]), "flipped_from": to_school(pick["flipped_from"])} for pick in row["picks"]]
    return len(ON3_PICKS)


def picks_as_of(player_id: str, day: date) -> list[dict] | None:
    """Each insider's pick as it stood on `day`: the current pick if made by then, else the one it replaced."""
    if player_id not in ON3_PICKS:
        return None
    standing = []
    for pick in ON3_PICKS[player_id]:
        made = date.fromisoformat(pick["date"]) if pick["date"][:4].isdigit() else None
        before = date.fromisoformat(pick["previous_date"]) if (pick.get("previous_date") or "")[:4] not in ("", "0001") else None
        if made and made <= day:
            standing.append(pick)
        elif before and before <= day and pick["flipped_from"]:
            standing.append({**pick, "school": pick["flipped_from"], "flipped_from": "", "confidence": pick.get("previous_confidence")})
    return standing


def on3_features(player_id: str, school_norm: str, day: date) -> dict:
    standing = picks_as_of(player_id, day)
    if standing is None:
        return {"on3_picks_other_school": None, "on3_switched_away": None, "on3_best_accuracy_other": None, "on3_max_confidence_other": None}
    other = [p for p in standing if p["school"] and not same_school(p["school"], school_norm)]
    return {"on3_picks_other_school": len(other),
            "on3_switched_away": sum(1 for p in other if p["flipped_from"] and same_school(p["flipped_from"], school_norm)),
            "on3_best_accuracy_other": max((float(p["expert_accuracy"] or 0) for p in other), default=0.0),
            "on3_max_confidence_other": max((float(p["confidence"] or 0) for p in other), default=0.0)}


def same_school(a: str, b: str) -> bool:
    return bool(a and b) and (a == b or a.startswith(b + " ") or b.startswith(a + " "))


def insider_features(player_id: str, school_norm: str, day: date) -> dict:
    names = [n for n in NUMERIC_FEATURES if n.startswith("insider_")]
    if not INSIDER or day < INSIDER_COVERAGE_START + timedelta(days=30):
        return dict.fromkeys(names)
    rows = [r for r in INSIDER.get(player_id, []) if day - timedelta(days=90) < r[0] <= day]
    recent = [r for r in rows if r[0] > day - timedelta(days=30)]
    last14 = sum(1 for r in rows if r[0] > day - timedelta(days=14))
    prior60 = sum(1 for r in INSIDER.get(player_id, []) if day - timedelta(days=74) < r[0] <= day - timedelta(days=14))
    other = lambda schools: any(not same_school(s, school_norm) for s in schools)
    national = lambda r: r[5] == "national"
    own = lambda r: same_school(r[5], school_norm)
    rival = lambda r: r[5] not in ("national", "unknown") and not own(r)
    return {"insider_articles_30d": len(recent),
            "insider_flip_talk_30d": sum(r[1] for r in recent),
            "insider_flip_talk_90d": sum(r[1] for r in rows),
            "insider_firm_talk_30d": sum(r[2] for r in recent),
            "insider_other_school_flip_30d": sum(1 for r in recent if r[1] and other(r[4])),
            "insider_other_school_prediction_90d": sum(1 for r in rows if other(r[7])),
            "insider_own_site_flip_90d": sum(1 for r in rows if r[1] and own(r)),
            "insider_national_flip_30d": sum(1 for r in recent if r[1] and national(r)),
            "insider_rival_site_flip_30d": sum(1 for r in recent if r[1] and rival(r)),
            "insider_decision_talk_30d": sum(r[6] for r in recent),
            "insider_decision_flip_30d": sum(1 for r in recent if r[6] and r[1]),
            "insider_articles_14d": last14,
            "insider_momentum": round(last14 - prior60 * 14 / 60, 2)}


def coaching_features(school_norm: str, start: date, day: date) -> dict:
    from fetch_coaching_changes import school_key
    changes = COACHING.get(SCHOOL_KEYS.get(school_norm, school_key(school_norm)), [])
    recent = [d for d in changes if day - timedelta(days=365) < d <= day]
    return {"coach_change_since_commit": int(any(start < d <= day for d in recent)),
            "days_since_coach_change": (day - recent[-1]).days if recent else NO_VISIT_DAYS}


def snapshot_rows(commitments: list[dict], cutoff: date, labeled_only: bool) -> list[dict]:
    crowd = crowding_index(commitments)
    classes = class_index(commitments)
    levels = program_levels(commitments)
    breaks = school_breaks(commitments)
    geo_cache: dict[tuple, dict] = {}
    rows = []
    for c in commitments:
        recruit, school_norm, start = c["recruit"], c["school_norm"], c["start"]
        class_year = c["class_year"]
        signing = early_signing_day(class_year)
        # Events before the commitment (the at-commit features) and after it, sorted by date.
        before = {"Commitment": 0, "Decommit": 0, "Offer": 0, "Official Visit": 0, "Unofficial Visit": 0}
        after: list[tuple[date, str, str, str]] = []
        for event in c["events"]:
            day = parse_date(event["event_date"])
            if not day:
                continue
            if day < start:
                if event["event_type"] in before:
                    before[event["event_type"]] += 1
            elif day > start:
                after.append((day, event["event_type"], norm_team(event["school"]), event.get("source", "247")))
        after.sort()
        after_days = [day for day, _, _, _ in after]

        geo_key = (recruit.get("high_school", ""), school_norm)
        if geo_key not in geo_cache:
            geo_cache[geo_key] = geography_features(*geo_key)
        geo = geo_cache[geo_key]

        last_grid = c["end"] - timedelta(days=1)
        if labeled_only:
            last_grid = min(last_grid, cutoff - timedelta(days=HORIZON_DAYS))
        else:
            last_grid = min(last_grid, cutoff)
        days = grid_dates(start, last_grid)
        if not labeled_only and last_grid >= start and (not days or days[-1] != last_grid):
            days.append(last_grid)  # the live board scores every commitment as of the cutoff
        for day in days:
            seen_with_source = after[: bisect_right(after_days, day)]
            seen = [(d, kind, s) for d, kind, s, _ in seen_with_source]
            elsewhere_ov = [(d, s) for d, t, s in seen if t == "Official Visit" and s != school_norm]
            elsewhere_uv = [(d, s) for d, t, s in seen if t == "Unofficial Visit" and s != school_norm]
            visits_elsewhere = elsewhere_ov + elsewhere_uv
            last_visit = max((d for d, _ in visits_elsewhere), default=None)
            position = recruit.get("position", "")
            peers = crowd[(class_year, school_norm, position)]
            my_rating = rating_float(recruit)
            active_peers = [p for p in peers if p[3] != c["key"][1] and p[0] <= day < p[1]]
            visited_geo = []
            for visited in {s for _, s in visits_elsewhere}:
                key = (recruit.get("high_school", ""), visited)
                if key not in geo_cache:
                    geo_cache[key] = geography_features(*key)
                visited_geo.append(geo_cache[key])
            visited_miles = [g["home_to_school_miles"] for g in visited_geo if isinstance(g.get("home_to_school_miles"), (int, float))]
            committed_miles = geo.get("home_to_school_miles")
            nearest_visited = min(visited_miles) if visited_miles else None
            recent = [b for b in breaks.get(school_norm, []) if day - timedelta(days=60) < b[0] <= day and b[2] != c["key"][1]]
            level = levels.get((school_norm, class_year))
            label = None
            if c["broke_on"] is not None and day < c["broke_on"] <= day + timedelta(days=HORIZON_DAYS):
                label = 1
            elif day + timedelta(days=HORIZON_DAYS) <= cutoff:
                label = 0
            # Before signing: does it break at any point before the signing period ends? Known once it has
            # broken, or once the commitment's at-risk window has closed.
            breaks_in_window = c["broke_on"] is not None and c["broke_on"] == c["end"]
            if breaks_in_window and c["broke_on"] > day:
                label_before_signing = 1
            elif not breaks_in_window and c["end"] <= cutoff:
                label_before_signing = 0
            else:
                label_before_signing = None
            rows.append({
                "class_year": class_year,
                "player_id": c["key"][1],
                "name": recruit.get("name", ""),
                "committed_school": c["school"],
                "commitment_date": start.isoformat(),
                "snapshot_date": day.isoformat(),
                "profile_url": recruit.get("profile_url", ""),
                "target": label,
                "target_before_signing": label_before_signing,
                "rating": my_rating,
                "prior_commitments_at_commit": before["Commitment"],
                "prior_decommits_at_commit": before["Decommit"],
                "offers_before_commit": before["Offer"],
                "official_visits_before_commit": before["Official Visit"],
                "unofficial_visits_before_commit": before["Unofficial Visit"],
                "home_to_school_miles": geo.get("home_to_school_miles"),
                "days_committed": (day - start).days,
                "days_to_signing": (signing - day).days,
                "commit_days_to_signing": (signing - start).days,
                "official_visits_elsewhere": len(elsewhere_ov),
                "official_visits_elsewhere_power": sum(s in POWER_TEAMS for _, s in elsewhere_ov),
                "unofficial_visits_elsewhere": len(elsewhere_uv),
                "official_visits_committed_school": sum(1 for _, t, s in seen if t == "Official Visit" and s == school_norm),
                "unofficial_visits_committed_school": sum(1 for _, t, s in seen if t == "Unofficial Visit" and s == school_norm),
                "offers_since_commit": sum(1 for _, t, _ in seen if t == "Offer"),
                "power_offers_since_commit": sum(1 for _, t, s in seen if t == "Offer" and s in POWER_TEAMS),
                "days_since_last_visit_elsewhere": (day - last_visit).days if last_visit else NO_VISIT_DAYS,
                "visits_elsewhere_last_30_days": sum(1 for d, _ in visits_elsewhere if (day - d).days <= 30),
                "other_schools_visited": len({s for _, s in visits_elsewhere}),
                "same_position_commits_now": len(active_peers),
                "higher_rated_same_position_now": sum(1 for p in active_peers if p[2] > my_rating),
                "class_commits_now": sum(1 for p in classes[(class_year, school_norm)] if p[2] != c["key"][1] and p[0] <= day < p[1]),
                "nearest_visited_elsewhere_miles": nearest_visited,
                "visited_school_miles_closer": (committed_miles - nearest_visited) if nearest_visited is not None and isinstance(committed_miles, (int, float)) else None,
                "most_visits_to_one_other_school": max((sum(1 for _, s in visits_elsewhere if s == school) for school in {s for _, s in visits_elsewhere}), default=0),
                "visited_home_state_school": int(any(str(g.get("is_in_state_commit")) == "yes" for g in visited_geo) and str(geo.get("is_in_state_commit")) != "yes"),
                **coaching_features(school_norm, start, day),
                **insider_features(c["key"][1], school_norm, day),
                **on3_features(c["key"][1], school_norm, day),
                "school_decommits_last_60_days": len(recent),
                "school_class_decommits_last_60_days": sum(1 for b in recent if b[1] == class_year),
                "program_rating_level": level,
                "rating_above_program": (my_rating - level) if level and my_rating else None,
                "position": position,
                "star_bucket": recruit.get("star_bucket", ""),
                "distance_bucket": str(geo.get("distance_bucket", "")),
                "is_in_state_commit": str(geo.get("is_in_state_commit", "")),
                "committed_power_team": "yes" if school_norm in POWER_TEAMS else "no",
                "visited_schools": ", ".join(sorted({s.title() for _, s in visits_elsewhere})),
                "web_report_events": "; ".join(f"{kind} {s.title()} {d:%-m/%-d}" for d, kind, s, source in seen_with_source if source == "web report"),
            })
    return rows


def numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def markdown(frame: pd.DataFrame) -> str:
    cells = frame.astype(object).where(frame.notna(), "")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + " --- |" * len(frame.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in cells.itertuples(index=False)]
    return "\n".join(lines)


def logistic_model() -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", CalibratedClassifierCV(LogisticRegression(max_iter=4000), method="sigmoid", cv=5))])


def boosted_model() -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
    ])
    categorical = [False] * len(NUMERIC_FEATURES) + [True] * len(CATEGORICAL_FEATURES)
    booster = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=5.0,
        categorical_features=categorical, early_stopping=True, validation_fraction=0.15, random_state=7)
    return Pipeline([("preprocess", preprocess), ("model", CalibratedClassifierCV(booster, method="isotonic", cv=3))])


def metrics(label: str, target: pd.Series, probability: pd.Series) -> dict:
    return {
        "sample": label,
        "snapshots": len(target),
        "breaks_within_60_days": int(target.sum()),
        "actual_rate": round(float(target.mean()), 4),
        "average_predicted": round(float(probability.mean()), 4),
        "auc": round(float(roc_auc_score(target, probability)), 4),
        "brier": round(float(brier_score_loss(target, probability)), 4),
        "log_loss": round(float(log_loss(target, probability.clip(1e-6, 1 - 1e-6))), 4),
    }


def board_backtest(frame: pd.DataFrame, column: str) -> dict:
    """On each grid date, rank that class's active commitments; how many of the top BOARD_SIZE broke
    within 60 days, vs the rate across the whole board that day."""
    hits, base, days = [], [], 0
    for _, group in frame.groupby("snapshot_date"):
        if len(group) < BOARD_SIZE * 4:
            continue
        top = group.nlargest(BOARD_SIZE, column)
        hits.append(top["target"].mean())
        base.append(group["target"].mean())
        days += 1
    return {"board_dates": days, "top_25_hit_rate": round(float(np.mean(hits)), 4) if hits else None,
            "board_base_rate": round(float(np.mean(base)), 4) if base else None,
            "lift": round(float(np.mean(hits) / np.mean(base)), 2) if hits and np.mean(base) else None}


def calibration(frame: pd.DataFrame, column: str) -> list[dict]:
    bins = [0, 0.02, 0.05, 0.10, 0.20, 0.35, 1.0]
    labels = ["0-2%", "2-5%", "5-10%", "10-20%", "20-35%", "35%+"]
    frame = frame.assign(bucket=pd.cut(frame[column], bins=bins, labels=labels, include_lowest=True))
    return [{"bucket": str(bucket), "snapshots": len(g), "predicted": round(float(g[column].mean()), 4),
             "actual": round(float(g["target"].mean()), 4)} for bucket, g in frame.groupby("bucket", observed=True)]


STAGES = ["after early signing day", "0-30 days out", "30-60 days out", "60-90 days out", "90-180 days out", "180+ days out"]
CALIBRATION_CLASSES = [2025, 2026]  # full cycles of held-out predictions


def stage(days_to_signing: pd.Series) -> pd.Series:
    return pd.cut(days_to_signing, [-9999, 0, 30, 60, 90, 180, 9999], labels=STAGES).astype(str)


def calibration_design(probability: pd.Series, days_to_signing: pd.Series) -> pd.DataFrame:
    p = probability.clip(1e-4, 1 - 1e-4).to_numpy()
    logit = np.log(p / (1 - p))
    stages = stage(days_to_signing).to_numpy()
    design = pd.DataFrame({"logit": logit})
    for name in STAGES:
        design[name] = (stages == name).astype(float)
        design[f"logit x {name}"] = design[name] * logit
    return design


def fit_calibrator(probability: pd.Series, days_to_signing: pd.Series, target: pd.Series) -> LogisticRegression:
    """Held-out predictions ran low late in the cycle (60-90 days out: 7.7% predicted, 12.5% actual), so raw
    scores are mapped to break rates separately for each stretch of the calendar. Within a stretch the
    mapping is increasing, so a board's order on any date doesn't change — only the percentages."""
    return LogisticRegression(max_iter=2000).fit(calibration_design(probability, days_to_signing), target)


# Coverage on a log scale for the blend: "more flip talk, more risk" keeps rising past the most extreme case
# in the history (the tree model flattens out there — Easton Royal has four times the flip talk of anyone
# in three years of data).
BLEND_FEATURES = ["insider_other_school_flip_30d", "insider_flip_talk_30d", "insider_other_school_prediction_90d",
                  "insider_decision_flip_30d", "insider_articles_30d", "official_visits_elsewhere", "visits_elsewhere_last_30_days"]


class Calibrator:
    """Maps the tree model's raw score to a break rate by stage of the cycle; with blend=True it also weighs
    coverage counts on a log scale. Fit only on held-out predictions."""

    def __init__(self, blend: bool):
        self.blend = blend
        self.model = LogisticRegression(max_iter=4000)

    def design(self, probability: pd.Series, rows: pd.DataFrame) -> pd.DataFrame:
        design = calibration_design(probability, rows["days_to_signing"])
        if self.blend:
            design["coverage_known"] = rows["insider_articles_30d"].notna().astype(float).to_numpy()
            for name in BLEND_FEATURES:
                design[f"log_{name}"] = np.log1p(pd.to_numeric(rows[name], errors="coerce").fillna(0).clip(lower=0).to_numpy())
            design["log_momentum_up"] = np.log1p(pd.to_numeric(rows["insider_momentum"], errors="coerce").fillna(0).clip(lower=0).to_numpy())
        return design

    def fit(self, probability: pd.Series, rows: pd.DataFrame, target: pd.Series) -> "Calibrator":
        self.model.fit(self.design(probability, rows), target)
        return self

    def predict(self, probability: pd.Series, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.design(probability, rows))[:, 1]


def blend_check(frame: pd.DataFrame, probability: str, target: str) -> tuple[pd.DataFrame, bool]:
    """Fit on one held-out class, check on the other, with and without the blend. The blend is used only if
    it ranks better on both checks combined and doesn't cost more than a point of top-25 hit rate."""
    rows, scores = [], {}
    for blend in (False, True):
        aucs, hits = [], []
        for fit_year, check_year in [(2025, 2026), (2026, 2025)]:
            fit, check = frame[frame["class_year"] == fit_year], frame[frame["class_year"] == check_year].copy()
            check["calibrated"] = Calibrator(blend).fit(fit[probability], fit, fit[target]).predict(check[probability], check)
            auc = float(roc_auc_score(check[target], check["calibrated"]))
            hit = board_backtest(check.assign(target=check[target]), "calibrated")["top_25_hit_rate"] or 0.0
            rows.append({"blend": blend, "fit_on": fit_year, "checked_on": check_year, "auc": round(auc, 4), "top_25_hit_rate": hit,
                         "predicted": round(float(check["calibrated"].mean()), 4), "actual": round(float(check[target].mean()), 4)})
            aucs.append(auc)
            hits.append(hit)
        scores[blend] = (float(np.mean(aucs)), float(np.mean(hits)))
    use = scores[True][0] > scores[False][0] and scores[True][1] >= scores[False][1] - 0.01
    return pd.DataFrame(rows), use


BOOTSTRAP_MODELS = 6
BLEND_RESULTS: list[tuple[pd.DataFrame, bool]] = []


def bootstrap_range(train: pd.DataFrame, target: str, live: pd.DataFrame, calibrator, full_model_prediction: pd.Series,
                    seed: int = 11) -> tuple[pd.Series, pd.Series, pd.Series]:
    """The median and 10th-90th percentile of calibrated predictions from the full-data model plus models
    trained on resampled commitments: the board's number, and how much it depends on which past
    commitments the model happened to learn from (wide for rare cases)."""
    rng = np.random.default_rng(seed)
    keys = train[["class_year", "player_id", "commitment_date"]].astype(str).agg("|".join, axis=1)
    codes, uniques = pd.factorize(keys)
    predictions = []
    for _ in range(BOOTSTRAP_MODELS):
        # Resample commitments as weights (how many times each was drawn), not duplicated rows: the model's
        # internal calibration splits rows into folds, and a duplicated commitment in two folds leaks.
        counts = np.bincount(rng.integers(0, len(uniques), size=len(uniques)), minlength=len(uniques))
        weight = counts[codes]
        sample = train[weight > 0]
        model = boosted_model()
        model.fit(sample[FEATURES], sample[target], model__sample_weight=weight[weight > 0])
        raw = pd.Series(model.predict_proba(live[FEATURES])[:, 1], index=live.index)
        predictions.append(calibrator.predict(raw, live))
    stacked = np.vstack(predictions + [full_model_prediction.to_numpy()])
    return (pd.Series(np.median(stacked, axis=0), index=live.index), pd.Series(np.percentile(stacked, 10, axis=0), index=live.index),
            pd.Series(np.percentile(stacked, 90, axis=0), index=live.index))


def before_signing_model(all_rows: pd.DataFrame, live: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, object]:
    """Chance a commitment breaks before the signing period ends — the question a staff asks — trained only
    on classes whose signing period is over (a live class only shows the commitments that already broke)."""
    frame = all_rows[all_rows["target_before_signing"].notna() & (all_rows["class_year"] <= max(CALIBRATION_CLASSES))].copy()
    frame["target_before_signing"] = frame["target_before_signing"].astype(int)
    rows = []
    for year in CALIBRATION_CLASSES:
        train, test_index = frame[frame["class_year"] < year], frame.index[frame["class_year"] == year]
        model = boosted_model()
        model.fit(train[FEATURES], train["target_before_signing"])
        frame.loc[test_index, "before_signing_oos"] = model.predict_proba(frame.loc[test_index, FEATURES])[:, 1]
    blend_df, use_blend = blend_check(frame, "before_signing_oos", "target_before_signing")
    blend_df.insert(0, "target", "before signing")
    BLEND_RESULTS.append((blend_df, use_blend))
    for fit_year, check_year in [(2025, 2026), (2026, 2025)]:
        fit, check = frame[frame["class_year"] == fit_year], frame[frame["class_year"] == check_year].copy()
        check["calibrated"] = Calibrator(use_blend).fit(fit["before_signing_oos"], fit, fit["target_before_signing"]).predict(check["before_signing_oos"], check)
        check["stage"] = stage(check["days_to_signing"])
        board = check.assign(target=check["target_before_signing"])
        rows.append({"checked_on": check_year, "stage": "all", "snapshots": len(check),
                     "auc": round(float(roc_auc_score(check["target_before_signing"], check["before_signing_oos"])), 4),
                     "calibrated": round(float(check["calibrated"].mean()), 4), "actual": round(float(check["target_before_signing"].mean()), 4),
                     **{f"top_{BOARD_SIZE}_hit_rate": board_backtest(board, "calibrated")["top_25_hit_rate"]}})
        for name, group in check.groupby("stage", sort=False):
            rows.append({"checked_on": check_year, "stage": name, "snapshots": len(group),
                         "auc": round(float(roc_auc_score(group["target_before_signing"], group["before_signing_oos"])), 4) if group["target_before_signing"].nunique() > 1 else None,
                         "calibrated": round(float(group["calibrated"].mean()), 4), "actual": round(float(group["target_before_signing"].mean()), 4)})
    held_out = frame[frame["class_year"].isin(CALIBRATION_CLASSES)]
    calibrator = Calibrator(use_blend).fit(held_out["before_signing_oos"], held_out, held_out["target_before_signing"])
    final = boosted_model()
    final.fit(frame[FEATURES], frame["target_before_signing"])
    raw = pd.Series(final.predict_proba(live[FEATURES])[:, 1], index=live.index)
    full = pd.Series(calibrator.predict(raw, live), index=live.index)
    probability, live["before_signing_low"], live["before_signing_high"] = bootstrap_range(frame, "target_before_signing", live, calibrator, full)
    return pd.DataFrame(rows), probability, calibrator


def reasons(row: pd.Series) -> str:
    out = []
    if row["official_visits_elsewhere"]:
        out.append(f"{int(row['official_visits_elsewhere'])} official visit(s) elsewhere since committing")
    elif row["unofficial_visits_elsewhere"]:
        out.append(f"{int(row['unofficial_visits_elsewhere'])} unofficial visit(s) elsewhere since committing")
    if row["visits_elsewhere_last_30_days"]:
        out.append("visited elsewhere in the last 30 days")
    if row["power_offers_since_commit"]:
        out.append(f"{int(row['power_offers_since_commit'])} power offer(s) since committing")
    if row["higher_rated_same_position_now"] >= 2:
        out.append(f"{int(row['higher_rated_same_position_now'])} higher-rated commits at his position")
    if pd.notna(row["on3_switched_away"]) and row["on3_switched_away"] >= 1:
        out.append(f"{int(row['on3_switched_away'])} On3 insider(s) switched their pick away from his school")
    elif pd.notna(row["on3_picks_other_school"]) and row["on3_picks_other_school"] >= 1:
        out.append(f"{int(row['on3_picks_other_school'])} On3 insider pick(s) for another school")
    if pd.notna(row["insider_own_site_flip_90d"]) and row["insider_own_site_flip_90d"] >= 1:
        out.append(f"his own school's beat writing about a flip ({int(row['insider_own_site_flip_90d'])} articles, 90 days)")
    if pd.notna(row["insider_decision_flip_30d"]) and row["insider_decision_flip_30d"] >= 1:
        out.append("decision talk alongside flip talk (30 days)")
    if pd.notna(row["insider_other_school_prediction_90d"]) and row["insider_other_school_prediction_90d"] >= 1:
        out.append(f"insiders predicting/favoring another school ({int(row['insider_other_school_prediction_90d'])} mentions, 90 days)")
    elif pd.notna(row["insider_flip_talk_30d"]) and row["insider_flip_talk_30d"] >= 2:
        out.append(f"flip talk in coverage ({int(row['insider_flip_talk_30d'])} articles, 30 days)")
    if row["coach_change_since_commit"]:
        out.append(f"his school's head coach left or was fired {int(row['days_since_coach_change'])} days ago")
    if row["most_visits_to_one_other_school"] >= 2:
        out.append(f"{int(row['most_visits_to_one_other_school'])} visits to the same other school")
    if row["visited_home_state_school"]:
        out.append("visited a home-state school")
    elif pd.notna(row["visited_school_miles_closer"]) and row["visited_school_miles_closer"] >= 150:
        out.append(f"visited a school {int(row['visited_school_miles_closer'])} miles closer to home")
    if row["school_decommits_last_60_days"] >= 3:
        out.append(f"{int(row['school_decommits_last_60_days'])} other decommits from this school in 60 days")
    # A reach commit: a clearly better prospect than the program usually lands (not a 5-star at Texas).
    if pd.notna(row["rating_above_program"]) and row["rating_above_program"] >= 0.04 and row["program_rating_level"] < 0.88:
        out.append("rated well above this program's usual recruit")
    if row["prior_decommits_at_commit"]:
        out.append("has decommitted before")
    if row["official_visits_committed_school"] and not row["official_visits_elsewhere"]:
        out.append("officially visited his committed school (stabilizing)")
    return "; ".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores-file", type=Path, default=Path("data/commitment_strength_scores.csv"))
    parser.add_argument("--timeline-file", type=Path, default=Path("data/timeline_events.csv"))
    parser.add_argument("--events-file", type=Path, default=Path("data/flip_model_events.csv"),
                        help="the at-commit model's scored events, used as the baseline")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--cutoff", help="data cutoff (YYYY-MM-DD); default: the last decommit in the timeline")
    parser.add_argument("--coaching-changes", type=Path, default=Path("data/coaching_changes.csv"))
    parser.add_argument("--insider-signals", type=Path, default=Path("data/insider_signals.csv"))
    parser.add_argument("--commit-dates", type=Path, default=Path("data/commit_dates.csv"))
    parser.add_argument("--extra-events", type=Path, default=Path("data/web_report_events.csv"),
                        help="visits/offers from the web reports (extract_web_report_events.py); rows with counted=yes are added to the timeline")
    args = parser.parse_args()

    recruits = {(r["class_year"], r["player_id"]): r for r in dedupe_commits(load_csv(args.scores_file)) if in_high_school_cycle(r, "committed_date")}
    timeline = [r for r in load_csv(args.timeline_file) if in_high_school_cycle(r)]
    web_events = [r for r in load_csv(args.extra_events) if r.get("counted") == "yes" and in_high_school_cycle(r)] if args.extra_events.exists() else []
    for row in web_events:
        timeline.append({"class_year": row["class_year"], "player_id": row["player_id"], "name": row["name"],
                         "event_type": row["event_type"], "event_date": row["event_date"], "school": row["school"], "source": "web report"})
    if web_events:
        print(f"added {len(web_events)} visits/offers from the web reports")
    picks = load_on3_picks(Path.home() / "Tars" / "reports" / "web_report" / "backfill")
    print(f"On3 insider picks for {picks:,} players" if picks else "no On3 insider picks yet (backfill_predictions.py)")
    signals = load_insider_signals(args.insider_signals)
    print(f"{signals:,} insider signals loaded" if signals else "no insider signals file; run extract_insider_signals.py")
    changes = load_coaching_changes(args.coaching_changes, args.commit_dates, timeline)
    print(f"{changes} head coaching changes loaded" if changes else "no coaching changes file; run fetch_coaching_changes.py")
    cutoff = date.fromisoformat(args.cutoff) if args.cutoff else max(parse_date(r["event_date"]) or date.min for r in timeline if r["event_type"] == "Decommit")
    grouped = events_by_player(timeline)
    commitments = build_commitments(recruits, grouped, cutoff)
    print(f"cutoff {cutoff}; {len(commitments)} commitments")

    frame = numeric_features(pd.DataFrame(snapshot_rows(commitments, cutoff, labeled_only=True)))
    frame = frame[frame["target"].notna()].copy()
    frame["target"] = frame["target"].astype(int)
    print(f"{len(frame)} labeled snapshots, {frame['target'].mean():.1%} break within {HORIZON_DAYS} days")

    # Baseline: the at-commit model's probability for the same commitment (out-of-sample for 2025+).
    baseline = pd.DataFrame(load_csv(args.events_file))[["class_year", "player_id", "commitment_event_date", "model_decommit_flip_probability"]]
    baseline["commitment_date"] = pd.to_datetime(baseline["commitment_event_date"], format="%m/%d/%Y").dt.date.astype(str)
    baseline["class_year"] = baseline["class_year"].astype(int)
    baseline = baseline.drop_duplicates(["class_year", "player_id", "commitment_date"])
    frame = frame.merge(baseline[["class_year", "player_id", "commitment_date", "model_decommit_flip_probability"]],
                        on=["class_year", "player_id", "commitment_date"], how="left")
    frame["at_commit_probability"] = pd.to_numeric(frame.pop("model_decommit_flip_probability"), errors="coerce")

    # Each test class is scored by models trained only on the classes before it.
    for year in TEST_YEARS:
        train = frame[frame["class_year"] < year]
        test_index = frame.index[frame["class_year"] == year]
        for name, build in [("logistic", logistic_model), ("boosted", boosted_model)]:
            model = build()
            model.fit(train[FEATURES], train["target"])
            frame.loc[test_index, f"{name}_probability"] = model.predict_proba(frame.loc[test_index, FEATURES])[:, 1]
            if year == TEST_YEARS[0]:
                frame.loc[train.index, f"{name}_train_probability"] = model.predict_proba(train[FEATURES])[:, 1]
        print(f"scored class {year} (trained on {len(train):,} snapshots)")

    models = ["at_commit_probability", "logistic_probability", "boosted_probability"]
    validation, boards, calib = [], [], []
    for year in TEST_YEARS:
        test = frame[frame["class_year"] == year]
        # Same rows for every model: the at-commit model only scored commitments with a known outcome.
        same = test[test["at_commit_probability"].notna()]
        for column in models:
            validation.append({"model": column.replace("_probability", ""), "rows": "same as at-commit", **metrics(f"class_{year}", same["target"], same[column])})
            boards.append({"model": column.replace("_probability", ""), "rows": "same as at-commit", "class_year": year, **board_backtest(same, column)})
        for column in models[1:]:
            validation.append({"model": column.replace("_probability", ""), "rows": "all snapshots", **metrics(f"class_{year}", test["target"], test[column])})
            boards.append({"model": column.replace("_probability", ""), "rows": "all snapshots", "class_year": year, **board_backtest(test, column)})
    test_all = frame[frame["class_year"].isin(TEST_YEARS)]
    for column in models[1:]:
        calib += [{"model": column.replace("_probability", ""), **row} for row in calibration(test_all, column)]
    first_train = frame[frame["class_year"] < TEST_YEARS[0]]
    for name in ["logistic", "boosted"]:
        validation.append({"model": name, "rows": "training data", **metrics(f"train_{first_train['class_year'].min()}_{TEST_YEARS[0] - 1}", first_train["target"], first_train[f"{name}_train_probability"])})

    # Recalibration by stage, checked on the class it wasn't fit to (fit 2025 → check 2026, and back).
    blend_df, use_blend = blend_check(frame, "boosted_probability", "target")
    blend_df.insert(0, "target", "next 60 days")
    BLEND_RESULTS.append((blend_df, use_blend))
    stage_rows = []
    for fit_year, check_year in [(2025, 2026), (2026, 2025)]:
        fit, check = frame[frame["class_year"] == fit_year], frame[frame["class_year"] == check_year].copy()
        check["calibrated"] = Calibrator(use_blend).fit(fit["boosted_probability"], fit, fit["target"]).predict(check["boosted_probability"], check)
        check["stage"] = stage(check["days_to_signing"])
        for name, group in check.groupby("stage", sort=False):
            stage_rows.append({"fit_on": fit_year, "checked_on": check_year, "stage": name, "snapshots": len(group),
                               "raw": round(float(group["boosted_probability"].mean()), 4),
                               "calibrated": round(float(group["calibrated"].mean()), 4),
                               "actual": round(float(group["target"].mean()), 4)})
    stage_df = pd.DataFrame(stage_rows)
    stage_df["stage"] = pd.Categorical(stage_df["stage"], STAGES[::-1], ordered=True)
    stage_df = stage_df.sort_values(["fit_on", "stage"])
    stage_df.to_csv(args.out_dir / "flip_snapshot_stage_calibration.csv", index=False)
    held_out = frame[frame["class_year"].isin(CALIBRATION_CLASSES)]
    calibrator = Calibrator(use_blend).fit(held_out["boosted_probability"], held_out, held_out["target"])

    validation_df, boards_df, calib_df = pd.DataFrame(validation), pd.DataFrame(boards), pd.DataFrame(calib)
    validation_df.to_csv(args.out_dir / "flip_snapshot_validation.csv", index=False)
    boards_df.to_csv(args.out_dir / "flip_snapshot_board_backtest.csv", index=False)
    calib_df.to_csv(args.out_dir / "flip_snapshot_calibration.csv", index=False)

    # Pick the better model on the test classes' AUC, refit on every labeled class, score live commits.
    test_auc = validation_df[validation_df["rows"] == "all snapshots"].groupby("model")["auc"].mean()
    best = "boosted" if test_auc.get("boosted", 0) > test_auc.get("logistic", 0) else "logistic"
    final = boosted_model() if best == "boosted" else logistic_model()
    final.fit(frame[FEATURES], frame["target"])
    live = numeric_features(pd.DataFrame(snapshot_rows(commitments, cutoff, labeled_only=False)))
    live = live[(live["snapshot_date"] == live.groupby(["class_year", "player_id", "commitment_date"])["snapshot_date"].transform("max"))]
    live = live[pd.to_datetime(live["snapshot_date"]).dt.date > cutoff - timedelta(days=GRID_DAYS)]
    live = live[live.apply(lambda r: date.fromisoformat(r["snapshot_date"]) < high_school_cycle_cutoff(int(r["class_year"])), axis=1)]
    live["model_raw_probability"] = final.predict_proba(live[FEATURES])[:, 1].round(4)
    full = pd.Series(calibrator.predict(live["model_raw_probability"], live), index=live.index)
    middle, low, high = bootstrap_range(frame, "target", live, calibrator, full)
    live["break_probability_60d"], live["break_60d_low"], live["break_60d_high"] = middle.round(4), low.round(4), high.round(4)
    print("scoring before-signing risk")
    all_rows = numeric_features(pd.DataFrame(snapshot_rows(commitments, cutoff, labeled_only=False)))
    before_df, before_probability, _ = before_signing_model(all_rows, live)
    before_df.to_csv(args.out_dir / "flip_snapshot_before_signing_validation.csv", index=False)
    live["break_before_signing"] = before_probability.round(4)
    live["before_signing_low"], live["before_signing_high"] = live["before_signing_low"].round(4), live["before_signing_high"].round(4)
    live["why"] = live.apply(reasons, axis=1)
    live = live.sort_values("break_before_signing", ascending=False)
    columns = ["class_year", "name", "position", "star_bucket", "committed_school", "commitment_date", "snapshot_date",
               "break_before_signing", "before_signing_low", "before_signing_high",
               "break_probability_60d", "break_60d_low", "break_60d_high", "why", "visited_schools", "web_report_events", "official_visits_elsewhere", "unofficial_visits_elsewhere",
               "power_offers_since_commit", "days_committed", "days_to_signing", "model_raw_probability", "rating", "player_id", "profile_url"]
    (args.out_dir / "flip_boards").mkdir(parents=True, exist_ok=True)
    live[columns].to_csv(args.out_dir / "flip_boards" / "live_flip_risk.csv", index=False)

    summary = [
        "# Flip snapshot model", "",
        f"- Data cutoff: {cutoff}. Snapshots every {GRID_DAYS} days of every active commitment; target = decommit/flip within {HORIZON_DAYS} days.",
        f"- Labeled snapshots: {len(frame):,} ({frame['target'].mean():.1%} break within {HORIZON_DAYS} days). Each test class ({', '.join(map(str, TEST_YEARS))}) is scored by models trained only on the classes before it.",
        f"- Live board model: {best} (higher test AUC), refit on all labeled snapshots; {len(live):,} active commitments scored as of {cutoff}.", "",
        "## Test classes", "", markdown(validation_df), "",
        f"## Weekly board check (top {BOARD_SIZE} on each date)", "", markdown(boards_df), "",
        "## Calibration (test classes, raw model)", "", markdown(calib_df), "",
        "## Recalibration by stage of the cycle (fit on one class, checked on the other)", "",
        "The live board's `break_probability_60d` is the raw score recalibrated this way, fit on both classes.", "",
        markdown(stage_df), "",
        "## Coverage blend (fit on one held-out class, checked on the other)", "",
        "; ".join(f"{df['target'].iloc[0]}: {'blend used' if used else 'blend not used (no improvement)'}" for df, used in BLEND_RESULTS), "",
        markdown(pd.concat([df for df, _ in BLEND_RESULTS])) if BLEND_RESULTS else "", "",
        "## Before signing: breaks at any point before the signing period ends", "",
        f"Trained on complete classes only; each checked on a model trained on earlier classes and recalibrated on the other. The live board's `break_before_signing` uses all of them. Ranges (`*_low`/`*_high`) are the 10th-90th percentile of {BOOTSTRAP_MODELS} models trained on resampled commitments.", "",
        markdown(before_df), "",
    ]
    if BLEND_RESULTS:
        pd.concat([df for df, _ in BLEND_RESULTS]).to_csv(args.out_dir / "flip_snapshot_blend_check.csv", index=False)
    (args.out_dir / "flip_snapshot_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
