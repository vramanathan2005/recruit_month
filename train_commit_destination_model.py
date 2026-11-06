#!/usr/bin/env python3
"""Where will an uncommitted recruit commit? Per-school odds for every uncommitted recruit in the live classes.

The commitment counterpart to train_flip_destination_model.py (which asks where a committed recruit goes if he
flips). For every past recruit's first commitment, this takes snapshots in the two years before he committed (as a board
would have seen him: every two weeks in the last 120 days, every four weeks before that) and lists his candidate schools as of that date: every school that had
offered him or hosted him on a visit. A model learns which candidate he committed to from what was known then:
how far from signing day it is, offers and how recent, official and unofficial visits and how recent, distance
from home, in-state, program level
(and how his rating compares), how crowded his position already was there, insider predictions naming the school,
and On3 insider picks.

Scores are shared out among a player's candidates, and a share is held back for "somewhere not on his radar yet"
(a school with no offer or visit on record yet): that share is learned too, small for a recruit with many offers
and official visits, large for one with few on record.

    .venv/bin/python train_commit_destination_model.py     (any time after the 247 scrape)

Outputs (data/):
- commit_destination_validation.csv     held-out classes: top-1 / top-3 accuracy, vs a most-visits rule
- commit_destination_calibration.csv    predicted vs actual share, held-out classes
- flip_boards/live_commit_destinations.csv   every uncommitted live recruit's candidate schools, odds and reasons
- flip_boards/live_commit_outlook.csv        one row per uncommitted recruit: top schools, Texas's odds, off-radar share
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import average_precision_score, roc_auc_score

from backtest_commitment_strength import events_by_player
from build_flip_boards import POWER_TEAMS
from geography_features import geography_features
from score_commitment_strength import dedupe_commits, early_signing_day, in_high_school_cycle, load_csv, norm_team, parse_date, rating_float
from train_flip_destination_model import school_of
from train_flip_snapshot_model import (INSIDER, INSIDER_COVERAGE_START, build_commitments, crowding_index, grid_dates,
                                       load_insider_signals, load_on3_picks, picks_as_of, program_levels)

# Snapshots from up to two years before each first commitment: every two weeks in the last 120 days, every four weeks
# before that. A live recruit can be a year or more from his decision, so the model has to have seen that far out.
LOOKBACK_DAYS = 730
NEAR_DAYS = 120
TRAIN_FROM = 2022
TEST_YEARS = [2025, 2026]
LIVE_CLASSES = [2027, 2028]
NEVER = 9999
TEXAS = norm_team("Texas")

CANDIDATE_NUMERIC = [
    "offered", "days_since_offer", "official_visits", "unofficial_visits", "visits_total", "days_since_last_visit",
    "visit_recency_rank", "official_visit_rank", "miles_from_home", "program_level", "rating_above_program",
    "position_commits_there_now", "candidates", "schools_officially_visited", "days_to_signing_day",
    "insider_prediction_mentions_90d", "insider_mentions_90d", "on3_picks_for_school",
]
CANDIDATE_CATEGORICAL = ["in_state", "power_program", "star_bucket"]
CANDIDATE_FEATURES = CANDIDATE_NUMERIC + CANDIDATE_CATEGORICAL
# "Is the school he commits to one of his candidates at all?" — per player-snapshot.
COVERAGE_FEATURES = ["rating", "candidates", "offers_on_record", "official_visits_on_record", "visits_on_record", "days_to_signing_day"]


def candidate_rows(recruit: dict, player_id: str, class_year: int, events: list[dict], day: date,
                   geo_cache: dict, levels: dict, crowd: dict) -> tuple[list[dict], dict]:
    """His candidate schools as of `day` (offers and visits dated on or before it), with features; plus context."""
    rating = rating_float(recruit)
    known = [(parse_date(e["event_date"]), e["event_type"], school_of(e)) for e in events]
    known = [(d, t, s) for d, t, s in known if d and d <= day and s and t in ("Offer", "Official Visit", "Unofficial Visit")]
    by_school: dict[str, list[tuple[date, str]]] = {}
    for d, t, s in known:
        by_school.setdefault(s, []).append((d, t))
    home = recruit.get("high_school", "")
    last_visit = {s: max((d for d, t in ev if "Visit" in t), default=None) for s, ev in by_school.items()}
    recency_rank = {s: rank for rank, (_, s) in enumerate(sorted(((d, s) for s, d in last_visit.items() if d), reverse=True), 1)}
    last_official = {s: max((d for d, t in ev if t == "Official Visit"), default=None) for s, ev in by_school.items()}
    official_rank = {s: rank for rank, (_, s) in enumerate(sorted(((d, s) for s, d in last_official.items() if d), reverse=True), 1)}
    officially_visited = sum(1 for d in last_official.values() if d)
    covered = bool(INSIDER) and day >= INSIDER_COVERAGE_START + timedelta(days=30)
    talk = [r for r in INSIDER.get(player_id, []) if day - timedelta(days=90) < r[0] <= day] if covered else []
    standing = picks_as_of(player_id, day)
    rows = []
    for s, ev in by_school.items():
        if (home, s) not in geo_cache:
            geo_cache[(home, s)] = geography_features(home, s)
        g = geo_cache[(home, s)]
        miles = g.get("home_to_school_miles") if isinstance(g.get("home_to_school_miles"), (int, float)) else None
        offers = [d for d, t in ev if t == "Offer"]
        level = levels.get((s, class_year))
        peers = crowd.get((class_year, s, recruit.get("position", "")), [])
        rows.append({
            "school": s,
            "offered": int(bool(offers)),
            "days_since_offer": (day - max(offers)).days if offers else NEVER,
            "official_visits": sum(1 for _, t in ev if t == "Official Visit"),
            "unofficial_visits": sum(1 for _, t in ev if t == "Unofficial Visit"),
            "visits_total": sum(1 for _, t in ev if "Visit" in t),
            "days_since_last_visit": (day - last_visit[s]).days if last_visit.get(s) else NEVER,
            "visit_recency_rank": recency_rank.get(s, NEVER),
            "official_visit_rank": official_rank.get(s, NEVER),
            "miles_from_home": miles,
            "program_level": level,
            "rating_above_program": (rating - level) if level and rating else None,
            "position_commits_there_now": sum(1 for p in peers if p[0] <= day < p[1]),
            "candidates": len(by_school),
            "schools_officially_visited": officially_visited,
            # How far out he is: far from signing day, more can change (offers to come, favorites to move)
            "days_to_signing_day": (early_signing_day(class_year) - day).days,
            # Coverage naming this school in a prediction ("the smart money is on the Tigers"), or at all
            "insider_prediction_mentions_90d": sum(1 for r in talk if s in r[7]) if covered else None,
            "insider_mentions_90d": sum(1 for r in talk if s in r[4]) if covered else None,
            "on3_picks_for_school": sum(1 for pick in standing if pick["school"] == s) if standing is not None else None,
            "in_state": str(g.get("is_in_state_commit", "")),
            "power_program": "yes" if s in POWER_TEAMS else "no",
            "star_bucket": recruit.get("star_bucket", ""),
        })
    context = {
        "rating": rating,
        "candidates": len(by_school),
        "offers_on_record": sum(1 for _, t, _ in known if t == "Offer"),
        "official_visits_on_record": sum(1 for _, t, _ in known if t == "Official Visit"),
        "visits_on_record": sum(1 for _, t, _ in known if "Visit" in t),
        "days_to_signing_day": (early_signing_day(class_year) - day).days,
    }
    return rows, context


def candidate_model() -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", "passthrough", CANDIDATE_NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CANDIDATE_CATEGORICAL),
    ])
    categorical = [False] * len(CANDIDATE_NUMERIC) + [True] * len(CANDIDATE_CATEGORICAL)
    booster = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=200,
                                             l2_regularization=5.0, categorical_features=categorical, random_state=7)
    return Pipeline([("preprocess", preprocess), ("model", CalibratedClassifierCV(booster, method="isotonic", cv=3))])


def share_out(frame: pd.DataFrame, score: str, coverage: pd.Series, alpha: float) -> pd.Series:
    """Candidate scores → P(commits to X): score^alpha normalized within each snapshot, times the snapshot's chance
    his school is one of his candidates at all (alpha fit on past classes: how much clear favorites should take)."""
    powered = frame[score].clip(lower=1e-6) ** alpha
    return powered / powered.groupby(frame["snapshot_id"]).transform("sum") * frame["snapshot_id"].map(coverage)


def fit_sharpness(frame: pd.DataFrame, score: str) -> float:
    covered = frame[frame.groupby("snapshot_id")["target"].transform("max") == 1]
    best, best_ll = 1.0, -np.inf
    for alpha in np.arange(0.5, 5.01, 0.1):
        powered = covered[score].clip(lower=1e-6) ** alpha
        share = powered / powered.groupby(covered["snapshot_id"]).transform("sum")
        ll = float(np.log(share[covered["target"] == 1].clip(lower=1e-9)).sum())
        if ll > best_ll:
            best, best_ll = float(alpha), ll
    return round(best, 2)


def most_visits_rule(frame: pd.DataFrame) -> pd.Series:
    """Baseline: most official visits, then most recent visit, then any offer."""
    return frame["official_visits"] * 1000 + frame["unofficial_visits"] * 100 - np.minimum(frame["days_since_last_visit"], NEVER) / 100 + frame["offered"]


def candidate_scores(target: pd.Series, score: pd.Series) -> dict:
    y = target.astype(int)
    if not 0 < y.sum() < len(y):
        return {"candidate_roc_auc": "", "candidate_pr_auc": "", "candidate_base_rate": ""}
    return {"candidate_roc_auc": round(float(roc_auc_score(y, score)), 4), "candidate_pr_auc": round(float(average_precision_score(y, score)), 4),
            "candidate_base_rate": round(float(y.mean()), 4)}


def evaluate(frame: pd.DataFrame, snapshots: pd.DataFrame, column: str, label: str) -> dict:
    ranked = frame.assign(rank=frame.groupby("snapshot_id")[column].rank(ascending=False, method="first"))
    ids = snapshots["snapshot_id"]
    top1 = ranked[ranked["rank"] <= 1].groupby("snapshot_id")["target"].max().reindex(ids).fillna(0)
    top3 = ranked[ranked["rank"] <= 3].groupby("snapshot_id")["target"].max().reindex(ids).fillna(0)
    covered = snapshots.set_index("snapshot_id")["destination_is_candidate"].reindex(ids).to_numpy() == 1
    before = snapshots.set_index("snapshot_id")["days_before_commit"].reindex(ids).to_numpy()
    near = before <= 30
    horizons = {f"top_3_{name}": round(float(top3[(before > lo) & (before <= hi)].mean()), 4)
                for name, lo, hi in (("0_30_days", 0, 30), ("31_120_days", 30, 120), ("4_12_months", 120, 365), ("1_2_years", 365, 9999))}
    return {"method": label, "snapshots": len(ids), **horizons,
            "top_1_all": round(float(top1.mean()), 4), "top_3_all": round(float(top3.mean()), 4),
            "top_1_when_on_radar": round(float(top1[covered].mean()), 4), "top_3_when_on_radar": round(float(top3[covered].mean()), 4),
            "top_1_last_30_days": round(float(top1[near].mean()), 4), "top_3_last_30_days": round(float(top3[near].mean()), 4),
            "destination_on_radar": round(float(covered.mean()), 4),
            # each candidate school as a yes/no (did he go there?), over every candidate in every snapshot
            **candidate_scores(frame["target"], frame[column])}


def first_commitments(grouped: dict) -> dict:
    """(class, player) → (date, school) of his first commitment."""
    out = {}
    for key, events in grouped.items():
        dated = sorted((parse_date(e["event_date"]), school_of(e)) for e in events if e["event_type"] == "Commitment" and parse_date(e["event_date"]))
        if dated and dated[0][1]:
            out[key] = dated[0]
    return out


def web_report_events(rows: list[dict], timeline: list[dict]) -> list[dict]:
    """Visits and offers from Tars's web reports that 247 doesn't have. Besides the ones counted outright, this takes
    those reported as having happened without an exact date ("visited Miami late last month"), dated to the article's
    report of them: they still say a school is recruiting him. Each is counted once, and not when 247 or an earlier
    article already has the same school within two weeks. Planned visits don't count."""
    same = lambda a, b: abs((a - b).days) <= 14
    known: dict[tuple, list[tuple]] = {}
    for r in timeline:
        d = parse_date(r["event_date"])
        if d:
            known.setdefault((r["player_id"], r["event_type"]), []).append((d, norm_team(school_of(r))))
    out = []
    for r in sorted(rows, key=lambda r: parse_date(r["event_date"]) or date.min):
        if not (r.get("counted") == "yes" or r.get("why_not_counted") == "date not stated exactly") or not r.get("player_id"):
            continue
        if r["event_type"] not in ("Offer", "Official Visit", "Unofficial Visit") or not in_high_school_cycle(r):
            continue
        d, school = parse_date(r["event_date"]), norm_team(school_of(r))
        seen = known.setdefault((r["player_id"], r["event_type"]), [])
        if not d or not school or any(same(d, k) and (school == s or school.startswith(s + " ") or s.startswith(school + " ")) for k, s in seen):
            continue
        seen.append((d, school))
        out.append({k: r[k] for k in ("class_year", "player_id", "name", "event_type", "event_date", "school")})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores-file", type=Path, default=Path("data/commitment_strength_scores.csv"))
    parser.add_argument("--recruits-file", type=Path, default=Path("data/commit_dates.csv"))
    parser.add_argument("--timeline-file", type=Path, default=Path("data/timeline_events.csv"))
    parser.add_argument("--extra-events", type=Path, default=Path("data/web_report_events.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    scored = {(r["class_year"], r["player_id"]): r for r in dedupe_commits(load_csv(args.scores_file)) if in_high_school_cycle(r, "committed_date")}
    recruits = {(r["class_year"], r["player_id"]): r for r in dedupe_commits(load_csv(args.recruits_file))}
    timeline = [r for r in load_csv(args.timeline_file) if in_high_school_cycle(r)]
    if args.extra_events.exists():
        timeline += web_report_events(load_csv(args.extra_events), timeline)
    # Today is today: 247 carries scheduled future visits and the odd bad date (an offer in 2035), and nothing dated
    # after today counts yet
    today = date.today()
    timeline = [r for r in timeline if (parse_date(r["event_date"]) or date.min) <= today]
    grouped = events_by_player(timeline)
    load_insider_signals(Path("data/insider_signals.csv"))
    load_on3_picks(Path.home() / "Tars" / "reports" / "web_report" / "backfill")
    commitments = build_commitments(scored, grouped, today)
    levels, crowd, geo_cache = program_levels(commitments), crowding_index(commitments), {}

    # Training: snapshots before each past recruit's first commitment
    firsts = first_commitments(grouped)
    candidates, snapshots = [], []
    for key, (committed_on, school) in firsts.items():
        class_year = int(key[0])
        recruit = recruits.get(key) or scored.get(key)
        if not recruit or not TRAIN_FROM <= class_year <= max(TEST_YEARS):
            continue
        days = grid_dates(committed_on - timedelta(days=LOOKBACK_DAYS), committed_on - timedelta(days=1))
        days = [d for i, d in enumerate(days) if (committed_on - d).days <= NEAR_DAYS or i % 2 == len(days) % 2]
        for day in days:
            rows, context = candidate_rows(recruit, key[1], class_year, grouped[key], day, geo_cache, levels, crowd)
            snapshot_id = f"{class_year}|{key[1]}|{day}"
            snapshots.append({"snapshot_id": snapshot_id, "class_year": class_year, "destination_is_candidate": int(any(r["school"] == school for r in rows)),
                              "days_before_commit": (committed_on - day).days, **context})
            for r in rows:
                candidates.append({"snapshot_id": snapshot_id, "class_year": class_year, "target": int(r["school"] == school), **r})
    frame, snapshots = pd.DataFrame(candidates), pd.DataFrame(snapshots)
    frame[CANDIDATE_NUMERIC] = frame[CANDIDATE_NUMERIC].apply(pd.to_numeric, errors="coerce")
    print(f"{len(snapshots):,} snapshots before {snapshots['snapshot_id'].str.rsplit('|', n=1).str[0].nunique():,} first commitments; "
          f"{len(frame):,} candidate rows; his school on his radar {snapshots['destination_is_candidate'].mean():.1%}")

    # Validation: train on the classes before, test on a held-out class
    validation, calibration, held = [], [], {}
    for year in TEST_YEARS:
        train, test = frame[frame["class_year"] < year], frame[frame["class_year"] == year].copy()
        s_train, s_test = snapshots[snapshots["class_year"] < year], snapshots[snapshots["class_year"] == year]
        model = candidate_model().fit(train[CANDIDATE_FEATURES], train["target"])
        test["score"] = model.predict_proba(test[CANDIDATE_FEATURES])[:, 1]
        cover = LogisticRegression(max_iter=3000).fit(s_train[COVERAGE_FEATURES], s_train["destination_is_candidate"])
        held[year] = (test, s_test, pd.Series(cover.predict_proba(s_test[COVERAGE_FEATURES])[:, 1], index=s_test["snapshot_id"].to_numpy()))
        print(f"  fit for {year}", flush=True)
    for year, (test, s_test, coverage) in held.items():
        alpha = fit_sharpness(pd.concat([held[y][0] for y in held if y != year]), "score")
        test["probability"] = share_out(test, "score", coverage, alpha)
        test["rule"] = most_visits_rule(test)
        validation.append({"class_year": year, "sharpness": alpha, **evaluate(test, s_test, "probability", "model")})
        validation.append({"class_year": year, **evaluate(test, s_test, "rule", "most-visits rule")})
        bins = [0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
        test["bucket"] = pd.cut(test["probability"], bins, labels=["0-5%", "5-10%", "10-20%", "20-35%", "35-50%", "50-75%", "75%+"], include_lowest=True)
        # Also by how far from signing day: a live 2028 recruit is over a year out, so the far-out rows are his check
        test["horizon"] = np.where(test["days_to_signing_day"] > 365, "over a year to signing day", "within a year of signing day")
        for (horizon, bucket), g in test.groupby(["horizon", "bucket"], observed=True):
            calibration.append({"class_year": year, "horizon": horizon, "bucket": str(bucket), "candidates": len(g),
                                "predicted": round(float(g["probability"].mean()), 4), "actual": round(float(g["target"].mean()), 4)})
    validation_df, calibration_df = pd.DataFrame(validation), pd.DataFrame(calibration)
    validation_df.to_csv(args.out_dir / "commit_destination_validation.csv", index=False)
    calibration_df.to_csv(args.out_dir / "commit_destination_calibration.csv", index=False)
    print(validation_df.to_string(index=False))
    print(calibration_df.to_string(index=False))

    # Live: every uncommitted recruit in the live classes, candidates as of today
    model = candidate_model().fit(frame[CANDIDATE_FEATURES], frame["target"])
    cover = LogisticRegression(max_iter=3000).fit(snapshots[COVERAGE_FEATURES], snapshots["destination_is_candidate"])
    alpha = fit_sharpness(pd.concat([h[0] for h in held.values()]), "score")
    live_rows, live_snaps = [], []
    for key, recruit in recruits.items():
        if int(key[0]) not in LIVE_CLASSES or recruit.get("committed_team", "").strip():
            continue
        rows, context = candidate_rows(recruit, key[1], int(key[0]), grouped.get(key, []), today, geo_cache, levels, crowd)
        snapshot_id = f"{key[0]}|{key[1]}"
        live_snaps.append({"snapshot_id": snapshot_id, "class_year": int(key[0]), "player_id": key[1], "name": recruit.get("name", ""),
                           "position": recruit.get("position", ""), "star_bucket": recruit.get("star_bucket", ""),
                           "high_school": recruit.get("high_school", ""), **{k: v for k, v in context.items() if k != "rating"}, "rating_value": context["rating"]})
        for r in rows:
            live_rows.append({"snapshot_id": snapshot_id, **r})
    live, live_snap = pd.DataFrame(live_rows), pd.DataFrame(live_snaps)
    live[CANDIDATE_NUMERIC] = live[CANDIDATE_NUMERIC].apply(pd.to_numeric, errors="coerce")
    live["score"] = model.predict_proba(live[CANDIDATE_FEATURES])[:, 1]
    coverage_frame = live_snap.rename(columns={"rating_value": "rating"})[COVERAGE_FEATURES].fillna(0)
    on_radar = pd.Series(cover.predict_proba(coverage_frame)[:, 1], index=live_snap["snapshot_id"].to_numpy())
    on_radar[live_snap.set_index("snapshot_id")["candidates"].reindex(on_radar.index).to_numpy() == 0] = 0.0   # no offer or visit on record
    live["commit_probability"] = share_out(live, "score", on_radar, alpha)

    def why(r) -> str:
        """Plain-English reasons for a school's number, in the wording Tars shows (export_for_tars passes them through)."""
        parts = []
        if pd.notna(r.on3_picks_for_school) and r.on3_picks_for_school:
            parts.append(f"{int(r.on3_picks_for_school)} On3 insider pick{'s' if r.on3_picks_for_school > 1 else ''}")
        if pd.notna(r.insider_prediction_mentions_90d) and r.insider_prediction_mentions_90d:
            parts.append(f"Insiders predict him here ({int(r.insider_prediction_mentions_90d)} mention{'s' if r.insider_prediction_mentions_90d > 1 else ''})")
        if r.official_visits:
            parts.append(f"{int(r.official_visits)} official visit{'s' if r.official_visits > 1 else ''}")
        if r.unofficial_visits:
            parts.append(f"{int(r.unofficial_visits)} unofficial visit{'s' if r.unofficial_visits > 1 else ''}")
        if r.days_since_last_visit < 60:
            parts.append("Visited in the last week" if r.days_since_last_visit < 7 else f"Last visited {int(r.days_since_last_visit)} days ago")
        if r.in_state == "yes":
            parts.append("In his home state")
        elif pd.notna(r.miles_from_home) and r.miles_from_home <= 200:
            parts.append(f"{int(r.miles_from_home)} miles from home")
        if r.offered and r.days_since_offer >= 180:
            months = int(r.days_since_offer // 30)
            parts.append(f"Offered early ({months} months ago)")
        elif r.offered and r.days_since_offer <= 30:
            parts.append("Offered in the last month")
        if r.top_program:
            parts.append("The highest-level program recruiting him")
        elif r.power_program == "yes" and pd.notna(r.rating_above_program) and r.rating_above_program >= -0.02:
            parts.append("Power program that signs players at his level")
        if r.position_commits_there_now >= 2:
            parts.append(f"Already has {int(r.position_commits_there_now)} commits at his position")
        if not parts and r.offered:
            parts.append("Offered")
        return "; ".join(parts)

    acronyms = {"lsu", "tcu", "ucf", "smu", "usc", "ucla", "byu", "unlv", "utsa", "utep", "uab", "fiu", "fau", "usf", "uconn", "umass", "nc state"}
    live["school_key"] = live["school"]
    live["school"] = live["school"].map(lambda s: s.upper() if s in acronyms else " ".join(w.upper() if w in acronyms else w.capitalize() for w in s.split()))
    # The strongest program on his list (by its recruits' average rating): in past classes, the usual pick
    live["top_program"] = live["program_level"].notna() & (live["program_level"] == live.groupby("snapshot_id")["program_level"].transform("max"))
    live["why"] = live.apply(why, axis=1)
    live = live.merge(live_snap[["snapshot_id", "class_year", "player_id", "name", "position"]], on="snapshot_id").sort_values(["snapshot_id", "commit_probability"], ascending=[True, False])
    out = live[["class_year", "player_id", "name", "position", "school", "school_key", "commit_probability", "why"]].copy()
    out["commit_probability"] = out["commit_probability"].round(4)
    (args.out_dir / "flip_boards").mkdir(exist_ok=True)
    out.to_csv(args.out_dir / "flip_boards" / "live_commit_destinations.csv", index=False)

    # One row per recruit: his top schools, Texas's odds, and the share not on his radar yet
    top = live.groupby("snapshot_id").head(3).groupby("snapshot_id").apply(
        lambda g: ", ".join(f"{s} {p:.0%}" for s, p in zip(g["school"], g["commit_probability"])), include_groups=False)
    leader = live.groupby("snapshot_id").head(1).set_index("snapshot_id")
    texas = live[live["school_key"] == TEXAS].set_index("snapshot_id")["commit_probability"]
    outlook = live_snap.copy()
    outlook["likely_schools"] = outlook["snapshot_id"].map(top).fillna("")
    outlook["leader"] = outlook["snapshot_id"].map(leader["school"]).fillna("")
    outlook["leader_probability"] = outlook["snapshot_id"].map(leader["commit_probability"]).fillna(0).round(4)
    outlook["texas_probability"] = outlook["snapshot_id"].map(texas).fillna(0).round(4)
    outlook["texas_on_radar"] = outlook["snapshot_id"].isin(texas.index)
    outlook["not_on_radar"] = (1 - outlook["snapshot_id"].map(on_radar)).round(4)
    outlook = outlook.drop(columns=["snapshot_id"]).sort_values(["class_year", "rating_value"], ascending=[True, False])
    outlook.to_csv(args.out_dir / "flip_boards" / "live_commit_outlook.csv", index=False)
    print(f"sharpness {alpha}; {len(outlook):,} uncommitted recruits, {len(out):,} candidate rows; "
          f"Texas on the radar for {int(outlook['texas_on_radar'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
