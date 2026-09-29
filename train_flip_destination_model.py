#!/usr/bin/env python3
"""Where would a committed recruit go if he flipped? Per-school odds for the live flip board.

For every past commitment that broke and ended up at another school, this takes snapshots in the 120 days
before the break (as a weekly board would have seen him) and lists his candidate schools as of that date:
every school that had offered him, hosted him on a visit, or had an earlier commitment from him. A model
learns which candidate he ended up at from what was known then — offers, official and unofficial visits
(before and since committing), how recently he visited, distance from home and against his current school,
in-state, program level, and how crowded his position already was there.

Scores are shared out among a player's candidates, and a share is held back for "somewhere not on his
radar yet": 41% of past flips went to a school with no offer or visit on record before the break (70% of
those offered only afterwards). That share is learned too — it's small for a 5-star who has taken official
visits elsewhere, large for a 3-star with few offers on record.

On the live board:  P(flips to X) = P(breaks before signing) x P(goes to another school | breaks) x P(X | goes).

    .venv/bin/python train_flip_destination_model.py     (after train_flip_snapshot_model.py)

Outputs (data/):
- flip_destination_validation.csv          held-out classes: top-1 / top-3 accuracy, vs a visits-first rule
- flip_destination_calibration.csv         predicted vs actual share, held-out classes
- flip_boards/live_flip_destinations.csv   every live commitment's candidate schools with odds and reasons
- flip_boards/live_flip_risk.csv           gains a `likely_destinations` column
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from backtest_commitment_strength import events_by_player
from build_flip_boards import POWER_TEAMS
from geography_features import geography_features
from score_commitment_strength import dedupe_commits, in_high_school_cycle, load_csv, norm_team, parse_date, rating_float
from train_flip_snapshot_model import (INSIDER, INSIDER_COVERAGE_START, GRID_DAYS, build_commitments, crowding_index, grid_dates,
                                       load_insider_signals, program_levels)

LOOKBACK_DAYS = 120
TEST_YEARS = [2025, 2026]
NEVER = 9999
ARRIVAL_TYPES = ("Commitment", "Signing", "Enrollment")

CANDIDATE_NUMERIC = [
    "offered", "days_since_offer", "official_visit_before_commit", "official_visits_since_commit",
    "unofficial_visits_since_commit", "visits_total", "days_since_last_visit", "earlier_commitment",
    "miles_from_home", "miles_closer_than_committed", "program_level", "rating_above_program",
    "position_commits_there_now", "candidates", "visit_recency_rank",
    "insider_flip_mentions_90d", "insider_prediction_mentions_90d",
]
CANDIDATE_CATEGORICAL = ["in_state", "power_program", "star_bucket"]
CANDIDATE_FEATURES = CANDIDATE_NUMERIC + CANDIDATE_CATEGORICAL
# "Is the destination one of his candidates at all?" — per player-snapshot.
COVERAGE_FEATURES = ["rating", "candidates", "offers_on_record", "official_visits_elsewhere_since_commit", "any_visit_elsewhere_since_commit", "days_to_break_or_cutoff_unknown"]


def destination(c: dict) -> str | None:
    """The next school he committed to, signed with or enrolled at after the break (not the one he left)."""
    after = sorted((parse_date(e["event_date"]), e["event_type"], norm_team(e["school"])) for e in c["events"]
                   if parse_date(e["event_date"]) and parse_date(e["event_date"]) >= c["broke_on"])
    return next((s for _, t, s in after if t in ARRIVAL_TYPES and s != c["school_norm"]), None)


def candidate_rows(c: dict, day: date, geo_cache: dict, levels: dict, crowd: dict) -> tuple[list[dict], dict]:
    """His candidate schools as of `day` (events dated on or before it), with features; plus player-level context."""
    school = c["school_norm"]
    recruit = c["recruit"]
    rating = rating_float(recruit)
    known = [(parse_date(e["event_date"]), e["event_type"], norm_team(e["school"])) for e in c["events"]]
    known = [(d, t, s) for d, t, s in known if d and d <= day and s and s != school]
    by_school: dict[str, list[tuple[date, str]]] = {}
    for d, t, s in known:
        if t in ("Offer", "Official Visit", "Unofficial Visit", "Commitment"):
            by_school.setdefault(s, []).append((d, t))
    home_key = recruit.get("high_school", "")
    if (home_key, school) not in geo_cache:
        geo_cache[(home_key, school)] = geography_features(home_key, school)
    committed_miles = geo_cache[(home_key, school)].get("home_to_school_miles")
    last_visit = {s: max((d for d, t in ev if "Visit" in t), default=None) for s, ev in by_school.items()}
    visited = sorted((d, s) for s, d in last_visit.items() if d)
    recency_rank = {s: rank for rank, (_, s) in enumerate(reversed(visited), 1)}
    covered = bool(INSIDER) and day >= INSIDER_COVERAGE_START + timedelta(days=30)
    talk = [r for r in INSIDER.get(c["key"][1], []) if day - timedelta(days=90) < r[0] <= day] if covered else []
    rows = []
    for s, ev in by_school.items():
        if (home_key, s) not in geo_cache:
            geo_cache[(home_key, s)] = geography_features(home_key, s)
        g = geo_cache[(home_key, s)]
        miles = g.get("home_to_school_miles") if isinstance(g.get("home_to_school_miles"), (int, float)) else None
        offers = [d for d, t in ev if t == "Offer"]
        level = levels.get((s, c["class_year"]))
        peers = crowd.get((c["class_year"], s, recruit.get("position", "")), [])
        rows.append({
            "school": s,
            "offered": int(bool(offers)),
            "days_since_offer": (day - max(offers)).days if offers else NEVER,
            "official_visit_before_commit": int(any(t == "Official Visit" and d < c["start"] for d, t in ev)),
            "official_visits_since_commit": sum(1 for d, t in ev if t == "Official Visit" and d > c["start"]),
            "unofficial_visits_since_commit": sum(1 for d, t in ev if t == "Unofficial Visit" and d > c["start"]),
            "visits_total": sum(1 for _, t in ev if "Visit" in t),
            "days_since_last_visit": (day - last_visit[s]).days if last_visit.get(s) else NEVER,
            "earlier_commitment": int(any(t == "Commitment" for _, t in ev)),
            "miles_from_home": miles,
            "miles_closer_than_committed": (committed_miles - miles) if miles is not None and isinstance(committed_miles, (int, float)) else None,
            "program_level": level,
            "rating_above_program": (rating - level) if level and rating else None,
            "position_commits_there_now": sum(1 for p in peers if p[0] <= day < p[1]),
            "candidates": len(by_school),
            "visit_recency_rank": recency_rank.get(s, NEVER),
            # Coverage naming this school alongside flip talk or a prediction ("smart money is on the Tigers").
            "insider_flip_mentions_90d": sum(1 for r in talk if r[1] and s in r[4]) if covered else None,
            "insider_prediction_mentions_90d": sum(1 for r in talk if r[3] and s in r[4]) if covered else None,
            "in_state": str(g.get("is_in_state_commit", "")),
            "power_program": "yes" if s in POWER_TEAMS else "no",
            "star_bucket": recruit.get("star_bucket", ""),
        })
    context = {
        "rating": rating,
        "candidates": len(by_school),
        "offers_on_record": sum(1 for _, t, _ in known if t == "Offer"),
        "official_visits_elsewhere_since_commit": sum(1 for d, t, _ in known if t == "Official Visit" and d > c["start"]),
        "any_visit_elsewhere_since_commit": int(any("Visit" in t and d > c["start"] for d, t, _ in known)),
        "days_to_break_or_cutoff_unknown": 0,  # same for every row; kept so live and training frames match
    }
    return rows, context


def candidate_model() -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", "passthrough", CANDIDATE_NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CANDIDATE_CATEGORICAL),
    ])
    categorical = [False] * len(CANDIDATE_NUMERIC) + [True] * len(CANDIDATE_CATEGORICAL)
    booster = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=100,
                                             l2_regularization=5.0, categorical_features=categorical, random_state=7)
    return Pipeline([("preprocess", preprocess), ("model", CalibratedClassifierCV(booster, method="isotonic", cv=3))])


def coverage_model() -> LogisticRegression:
    return LogisticRegression(max_iter=3000)


def share_out(frame: pd.DataFrame, score: str, coverage: pd.Series) -> pd.Series:
    """Candidate scores → P(X | goes elsewhere): normalized within each snapshot, times the snapshot's
    chance the destination is one of his candidates at all."""
    total = frame.groupby("snapshot_id")[score].transform("sum")
    return frame[score] / total * frame["snapshot_id"].map(coverage)


def build_training(commitments: list[dict], cutoff: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    levels, crowd, geo_cache = program_levels(commitments), crowding_index(commitments), {}
    candidates, snapshots = [], []
    for c in commitments:
        if not (c["broke_on"] and c["broke_on"] == c["end"]):
            continue
        dest = destination(c)
        if not dest:
            continue
        start = max(c["start"], c["broke_on"] - timedelta(days=LOOKBACK_DAYS))
        for day in grid_dates(start, c["broke_on"] - timedelta(days=1)):
            rows, context = candidate_rows(c, day, geo_cache, levels, crowd)
            snapshot_id = f"{c['class_year']}|{c['key'][1]}|{c['start']}|{day}"
            in_candidates = any(r["school"] == dest for r in rows)
            snapshots.append({"snapshot_id": snapshot_id, "class_year": c["class_year"], "destination_is_candidate": int(in_candidates),
                              "days_before_break": (c["broke_on"] - day).days, **context})
            for r in rows:
                candidates.append({"snapshot_id": snapshot_id, "class_year": c["class_year"], "name": c["recruit"].get("name", ""),
                                   "committed_school": c["school"], "target": int(r["school"] == dest), **r})
    frame = pd.DataFrame(candidates)
    frame[CANDIDATE_NUMERIC] = frame[CANDIDATE_NUMERIC].apply(pd.to_numeric, errors="coerce")
    return frame, pd.DataFrame(snapshots)


def visits_first_rule(frame: pd.DataFrame) -> pd.Series:
    """Baseline: most official visits since committing, then most recent visit, then earliest offer."""
    return (frame["official_visits_since_commit"] * 1000 + frame["unofficial_visits_since_commit"] * 100
            - np.minimum(frame["days_since_last_visit"], NEVER) / 100 + frame["offered"])


def evaluate(frame: pd.DataFrame, snapshots: pd.DataFrame, column: str, label: str) -> dict:
    ranked = frame.assign(rank=frame.groupby("snapshot_id")[column].rank(ascending=False, method="first"))
    hit1 = ranked[ranked["rank"] <= 1].groupby("snapshot_id")["target"].max()
    hit3 = ranked[ranked["rank"] <= 3].groupby("snapshot_id")["target"].max()
    ids = snapshots["snapshot_id"]
    top1 = hit1.reindex(ids).fillna(0)
    top3 = hit3.reindex(ids).fillna(0)
    covered = snapshots.set_index("snapshot_id")["destination_is_candidate"].reindex(ids)
    return {"method": label, "snapshots": len(ids),
            "top_1_all": round(float(top1.mean()), 4), "top_3_all": round(float(top3.mean()), 4),
            "top_1_when_on_radar": round(float(top1[covered.to_numpy() == 1].mean()), 4),
            "top_3_when_on_radar": round(float(top3[covered.to_numpy() == 1].mean()), 4),
            "destination_on_radar": round(float(covered.mean()), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores-file", type=Path, default=Path("data/commitment_strength_scores.csv"))
    parser.add_argument("--timeline-file", type=Path, default=Path("data/timeline_events.csv"))
    parser.add_argument("--extra-events", type=Path, default=Path("data/web_report_events.csv"))
    parser.add_argument("--board", type=Path, default=Path("data/flip_boards/live_flip_risk.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    recruits = {(r["class_year"], r["player_id"]): r for r in dedupe_commits(load_csv(args.scores_file)) if in_high_school_cycle(r, "committed_date")}
    timeline = [r for r in load_csv(args.timeline_file) if in_high_school_cycle(r)]
    if args.extra_events.exists():
        timeline += [{k: r[k] for k in ("class_year", "player_id", "name", "event_type", "event_date", "school")}
                     for r in load_csv(args.extra_events) if r.get("counted") == "yes" and in_high_school_cycle(r)]
    cutoff = max(parse_date(r["event_date"]) or date.min for r in timeline if r["event_type"] == "Decommit")
    load_insider_signals(Path("data/insider_signals.csv"))
    commitments = build_commitments(recruits, events_by_player(timeline), cutoff)

    frame, snapshots = build_training(commitments, cutoff)
    frame = frame[frame["class_year"] <= max(TEST_YEARS)]
    snapshots = snapshots[snapshots["class_year"] <= max(TEST_YEARS)]
    print(f"{snapshots['snapshot_id'].nunique():,} snapshots before past breaks; "
          f"{len(frame):,} candidate rows; destination on his radar {snapshots['destination_is_candidate'].mean():.1%}")

    validation, calibration = [], []
    for year in TEST_YEARS:
        train, test = frame[frame["class_year"] < year], frame[frame["class_year"] == year].copy()
        s_train, s_test = snapshots[snapshots["class_year"] < year], snapshots[snapshots["class_year"] == year]
        model = candidate_model().fit(train[CANDIDATE_FEATURES], train["target"])
        test["score"] = model.predict_proba(test[CANDIDATE_FEATURES])[:, 1]
        cover = coverage_model().fit(s_train[COVERAGE_FEATURES], s_train["destination_is_candidate"])
        coverage = pd.Series(cover.predict_proba(s_test[COVERAGE_FEATURES])[:, 1], index=s_test["snapshot_id"].to_numpy())
        test["probability"] = share_out(test, "score", coverage)
        test["rule"] = visits_first_rule(test)
        validation.append({"class_year": year, **evaluate(test, s_test, "probability", "model")})
        validation.append({"class_year": year, **evaluate(test, s_test, "rule", "visits-first rule")})
        bins = [0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0]
        test["bucket"] = pd.cut(test["probability"], bins, labels=["0-5%", "5-10%", "10-20%", "20-35%", "35-50%", "50%+"], include_lowest=True)
        for bucket, g in test.groupby("bucket", observed=True):
            calibration.append({"class_year": year, "bucket": str(bucket), "candidates": len(g),
                                "predicted": round(float(g["probability"].mean()), 4), "actual": round(float(g["target"].mean()), 4)})
        print(f"tested on {year}")
    validation_df, calibration_df = pd.DataFrame(validation), pd.DataFrame(calibration)
    validation_df.to_csv(args.out_dir / "flip_destination_validation.csv", index=False)
    calibration_df.to_csv(args.out_dir / "flip_destination_calibration.csv", index=False)
    print(validation_df.to_string(index=False))
    print(calibration_df.to_string(index=False))

    # Live: every commitment on the board, candidates as of the cutoff.
    model = candidate_model().fit(frame[CANDIDATE_FEATURES], frame["target"])
    cover = coverage_model().fit(snapshots[COVERAGE_FEATURES], snapshots["destination_is_candidate"])
    breaks = [c for c in commitments if c["class_year"] <= max(TEST_YEARS) and c["broke_on"] and c["broke_on"] == c["end"]]
    goes_elsewhere = sum(destination(c) is not None for c in breaks) / len(breaks)
    board = pd.read_csv(args.board)
    by_key = {(c["class_year"], c["key"][1], c["start"].isoformat()): c for c in commitments}
    levels, crowd, geo_cache = program_levels(commitments), crowding_index(commitments), {}
    live_rows, live_snapshots, seen = [], [], set()
    for row in board.itertuples():
        c = by_key.get((int(row.class_year), str(row.player_id), row.commitment_date))
        if not c or (row.class_year, row.player_id, row.commitment_date) in seen:
            continue
        seen.add((row.class_year, row.player_id, row.commitment_date))
        rows, context = candidate_rows(c, cutoff, geo_cache, levels, crowd)
        snapshot_id = f"{row.class_year}|{row.player_id}|{row.commitment_date}"
        live_snapshots.append({"snapshot_id": snapshot_id, **context})
        for r in rows:
            live_rows.append({"snapshot_id": snapshot_id, "class_year": row.class_year, "name": row.name, "position": row.position,
                              "committed_school": row.committed_school, "break_before_signing": row.break_before_signing, **r})
    live, live_snap = pd.DataFrame(live_rows), pd.DataFrame(live_snapshots)
    live[CANDIDATE_NUMERIC] = live[CANDIDATE_NUMERIC].apply(pd.to_numeric, errors="coerce")
    live["score"] = model.predict_proba(live[CANDIDATE_FEATURES])[:, 1]
    coverage = pd.Series(cover.predict_proba(live_snap[COVERAGE_FEATURES])[:, 1], index=live_snap["snapshot_id"].to_numpy())
    live["if_he_flips"] = share_out(live, "score", coverage)
    live["flip_to_school"] = live["break_before_signing"] * goes_elsewhere * live["if_he_flips"]

    def why(r) -> str:
        parts = []
        if r.official_visits_since_commit:
            parts.append(f"{int(r.official_visits_since_commit)} official visit(s) since committing")
        elif r.official_visit_before_commit:
            parts.append("official visit before committing")
        if r.unofficial_visits_since_commit:
            parts.append(f"{int(r.unofficial_visits_since_commit)} unofficial visit(s) since committing")
        if r.days_since_last_visit < 60:
            parts.append(f"visited {int(r.days_since_last_visit)} days ago")
        if r.in_state == "yes":
            parts.append("in-state")
        elif pd.notna(r.miles_closer_than_committed) and r.miles_closer_than_committed >= 150:
            parts.append(f"{int(r.miles_closer_than_committed)} miles closer to home")
        if r.earlier_commitment:
            parts.append("was committed there before")
        if pd.notna(r.insider_prediction_mentions_90d) and r.insider_prediction_mentions_90d:
            parts.append(f"insiders favor/predict ({int(r.insider_prediction_mentions_90d)} mentions)")
        elif pd.notna(r.insider_flip_mentions_90d) and r.insider_flip_mentions_90d:
            parts.append(f"named in flip talk ({int(r.insider_flip_mentions_90d)} mentions)")
        if not parts and r.offered:
            parts.append("offered")
        return "; ".join(parts)

    acronyms = {"lsu", "tcu", "ucf", "smu", "usc", "ucla", "byu", "unlv", "utsa", "utep", "uab", "fiu", "fau", "usf", "uconn", "umass", "nc state"}
    live["school"] = live["school"].map(lambda s: s.upper() if s in acronyms else " ".join(w.upper() if w in acronyms else w.capitalize() for w in s.split()))
    live["why"] = live.apply(why, axis=1)
    live = live.sort_values(["break_before_signing", "snapshot_id", "if_he_flips"], ascending=[False, True, False])
    out = live[["class_year", "name", "position", "committed_school", "break_before_signing", "school", "if_he_flips", "flip_to_school", "why"]].copy()
    out[["if_he_flips", "flip_to_school"]] = out[["if_he_flips", "flip_to_school"]].round(4)
    out.to_csv(args.out_dir / "flip_boards" / "live_flip_destinations.csv", index=False)

    # Top three schools on the main board: "LSU 41%, Florida 22%, not on his radar yet 14%" (if he flips).
    radar = 1 - live_snap.set_index("snapshot_id").index.map(coverage).to_series(index=live_snap["snapshot_id"])
    top = live.groupby("snapshot_id").head(3).groupby("snapshot_id").apply(
        lambda g: ", ".join(f"{s} {p:.0%}" for s, p in zip(g["school"], g["if_he_flips"])), include_groups=False)
    board["snapshot_id"] = board["class_year"].astype(str) + "|" + board["player_id"].astype(str) + "|" + board["commitment_date"]
    board["likely_destinations"] = board["snapshot_id"].map(top).fillna("") + board["snapshot_id"].map(
        lambda s: f"; not on his radar yet {radar.get(s, 1.0):.0%}" if s in radar.index else "not on his radar yet 100%")
    board.drop(columns="snapshot_id").to_csv(args.board, index=False)
    print(f"goes to another school after a break: {goes_elsewhere:.1%}; wrote {len(out):,} live candidate rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
