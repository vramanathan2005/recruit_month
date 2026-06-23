#!/usr/bin/env python3
"""Backtest commitment vulnerability against 247 timeline outcomes.

This evaluates commitment *events*, not just each recruit's final commitment.
That matters because final commitments are often already the stable endpoint;
the useful historical question is whether a commitment looked vulnerable at the
time it was made and later decommitted/flipped.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from geography_features import geography_features
from score_commitment_strength import (
    build_position_context,
    clamp,
    dedupe_commits,
    early_signing_day,
    in_high_school_cycle,
    load_csv,
    norm_team,
    parse_date,
    rating_float,
    write_csv,
)


BUCKETS = [
    ("Weak / Attack", 0, 49.999),
    ("Vulnerable", 50, 64.999),
    ("Stable / Monitor", 65, 79.999),
    ("Strong", 80, 100),
]


def bucket_for(score: float) -> str:
    for label, low, high in BUCKETS:
        if low <= score <= high:
            return label
    return "Unknown"


def event_sort_key(event: dict[str, str]) -> tuple[date, int]:
    event_date = parse_date(event["event_date"]) or date.min
    priority = {"Decommit": 0, "Commitment": 1, "Signing": 2, "Enrollment": 3}.get(event["event_type"], 9)
    return event_date, priority


def events_by_player(events: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str, str, str, str]] = set()
    for event in events:
        if not in_high_school_cycle(event):
            continue
        event_key = (
            event.get("class_year", ""),
            event.get("player_id", ""),
            event.get("event_type", ""),
            event.get("event_date", ""),
            event.get("school", ""),
        )
        if event_key in seen:
            continue
        seen.add(event_key)
        grouped[(event["class_year"], event["player_id"])].append(event)
    for key in grouped:
        grouped[key].sort(key=event_sort_key)
    return grouped


def same_school(a: str, b: str) -> bool:
    return bool(a and b and norm_team(a) == norm_team(b))


def classify_outcome(commit_event: dict[str, str], player_events: list[dict[str, str]]) -> tuple[str, str, str]:
    commit_date = parse_date(commit_event["event_date"])
    commit_school = commit_event["school"]
    if not commit_date:
        return "unknown", "", ""

    future = [
        event
        for event in player_events
        if (parse_date(event["event_date"]) or date.min) > commit_date
    ]

    stable_event: tuple[str, str, str] | None = None
    for event in future:
        event_type = event["event_type"]
        event_school = event["school"]
        if event_type == "Decommit" and same_school(event_school, commit_school):
            return "decommitted", event["event_date"], event_school
        if event_type == "Commitment" and not same_school(event_school, commit_school):
            return "flipped", event["event_date"], event_school
        if event_type in {"Signing", "Enrollment"}:
            if not same_school(event_school, commit_school):
                return "flipped", event["event_date"], event_school
            if stable_event is None:
                stable_event = ("stuck", event["event_date"], event_school)

    return stable_event if stable_event else ("open", "", "")


def score_commitment_event(
    recruit: dict[str, str],
    commit_event: dict[str, str],
    player_events: list[dict[str, str]],
    position_counts: Counter,
    position_groups: dict[tuple[str, str, str], list[dict[str, str]]],
) -> dict[str, object]:
    commit_date = parse_date(commit_event["event_date"])
    class_year = int(commit_event["class_year"])
    commit_team = norm_team(commit_event["school"])

    prior_commitments = 0
    prior_decommits = 0
    offers_before_event = 0
    official_visits_before_event = 0
    unofficial_visits_before_event = 0
    for event in player_events:
        event_date = parse_date(event["event_date"])
        if not event_date or not commit_date or event_date >= commit_date:
            continue
        prior_commitments += event["event_type"] == "Commitment"
        prior_decommits += event["event_type"] == "Decommit"
        offers_before_event += event["event_type"] == "Offer"
        official_visits_before_event += event["event_type"] == "Official Visit"
        unofficial_visits_before_event += event["event_type"] == "Unofficial Visit"

    signing_day = early_signing_day(class_year)
    days_to_signing = (signing_day - commit_date).days if commit_date else 0
    if days_to_signing > 365:
        early_commit_risk = 45
    elif days_to_signing > 180:
        early_commit_risk = 25
    elif days_to_signing > 90:
        early_commit_risk = 20
    elif days_to_signing > 30:
        early_commit_risk = 10
    elif days_to_signing > 0:
        early_commit_risk = 3
    else:
        early_commit_risk = 0

    pos_key = (commit_event["class_year"], commit_team, recruit.get("position", ""))
    same_position_commits = max(position_counts.get(pos_key, 0) - 1, 0)
    current_rating = rating_float(recruit)
    higher_rated_same_position = 0
    for teammate in position_groups.get(pos_key, []):
        if teammate["player_id"] != recruit["player_id"] and rating_float(teammate) > current_rating:
            higher_rated_same_position += 1

    # In the historical event-level data, first commitments are riskier than
    # later recommitments. Later commitments are often closer to signing day.
    history_risk = 10 if prior_commitments == 0 else -5
    if prior_decommits:
        history_risk -= 5

    process_risk = 0
    if offers_before_event == 0:
        process_risk += 2
    elif offers_before_event >= 16:
        process_risk -= 4
    elif offers_before_event >= 6:
        process_risk -= 1
    if official_visits_before_event:
        process_risk -= 4
    elif unofficial_visits_before_event:
        process_risk += 2

    # Being the only player at a position reads more vulnerable historically.
    # Position crowding in this scrape is more of a school/class-strength signal
    # than a "the kid is getting pushed out" signal.
    if same_position_commits == 0:
        crowding_risk = 10
    elif same_position_commits == 1:
        crowding_risk = 5
    elif same_position_commits >= 3:
        crowding_risk = -3
    else:
        crowding_risk = 0

    geo = geography_features(recruit.get("high_school", ""), commit_team)
    distance_bucket = geo["distance_bucket"]
    if distance_bucket == "cross_country":
        distance_risk = 8
    elif distance_bucket == "far":
        distance_risk = 5
    elif distance_bucket == "regional":
        distance_risk = 2
    elif distance_bucket == "in_state":
        distance_risk = -2
    else:
        distance_risk = 0

    star_bucket = recruit.get("star_bucket", "")
    attention_risk = 7 if star_bucket == "5-star" else 4 if star_bucket == "4-star" else 0

    total_risk = clamp(
        early_commit_risk + history_risk + process_risk + crowding_risk + distance_risk + attention_risk,
        0,
        100,
    )
    strength = round(100 - total_risk, 1)
    outcome, outcome_date, outcome_school = classify_outcome(commit_event, player_events)

    return {
        "class_year": commit_event["class_year"],
        "player_id": commit_event["player_id"],
        "name": commit_event["name"],
        "position": recruit.get("position", ""),
        "high_school": recruit.get("high_school", ""),
        "rating": recruit.get("rating", ""),
        "star_bucket": recruit.get("star_bucket", ""),
        "committed_school_at_event": commit_event["school"],
        "commitment_event_date": commit_event["event_date"],
        "event_strength_score": strength,
        "event_vulnerability_score": round(total_risk, 1),
        "event_strength_bucket": bucket_for(strength),
        "outcome": outcome,
        "outcome_date": outcome_date,
        "outcome_school": outcome_school,
        "is_decommit_or_flip": "yes" if outcome in {"decommitted", "flipped"} else "no",
        "days_to_early_signing_day": days_to_signing if commit_date else "",
        "prior_commitments_at_event": prior_commitments,
        "prior_decommits_at_event": prior_decommits,
        "offers_before_event": offers_before_event,
        "official_visits_before_event": official_visits_before_event,
        "unofficial_visits_before_event": unofficial_visits_before_event,
        "same_position_commits_at_school": same_position_commits,
        "higher_rated_same_position_commits": higher_rated_same_position,
        "early_commit_risk": round(early_commit_risk, 1),
        "history_risk": round(history_risk, 1),
        "process_risk": round(process_risk, 1),
        "class_crowding_risk": round(crowding_risk, 1),
        **geo,
        "distance_risk": distance_risk,
        "attention_risk": attention_risk,
    }


def summarize_bucket(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for label, _, _ in BUCKETS:
        bucket_rows = [row for row in rows if row["event_strength_bucket"] == label and row["outcome"] != "open"]
        unstable = sum(row["outcome"] in {"decommitted", "flipped"} for row in bucket_rows)
        total = len(bucket_rows)
        out.append(
            {
                "event_strength_bucket": label,
                "commitment_events_with_known_outcome": total,
                "decommit_or_flip_events": unstable,
                "decommit_or_flip_rate": round(unstable / total, 4) if total else "",
                "stick_rate": round(1 - unstable / total, 4) if total else "",
            }
        )
    return out


def summarize_by_position(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["outcome"] != "open":
            grouped[str(row["position"])].append(row)
    out = []
    for position, position_rows in sorted(grouped.items()):
        if len(position_rows) < 30:
            continue
        unstable = sum(row["outcome"] in {"decommitted", "flipped"} for row in position_rows)
        out.append(
            {
                "position": position,
                "known_commitment_events": len(position_rows),
                "decommit_or_flip_events": unstable,
                "decommit_or_flip_rate": round(unstable / len(position_rows), 4),
            }
        )
    return sorted(out, key=lambda row: row["decommit_or_flip_rate"], reverse=True)


def markdown_summary(event_rows: list[dict[str, object]], bucket_rows: list[dict[str, object]]) -> str:
    known = [row for row in event_rows if row["outcome"] != "open"]
    unstable = [row for row in known if row["outcome"] in {"decommitted", "flipped"}]
    by_outcome = Counter(str(row["outcome"]) for row in event_rows)

    lines = ["# Commitment Strength Backtest", ""]
    lines.append(f"- Commitment events tested: {len(event_rows)}")
    lines.append(f"- Known outcomes: {len(known)}")
    lines.append(f"- Decommit/flip outcomes: {len(unstable)}")
    lines.append(f"- Overall decommit/flip rate: {len(unstable) / len(known):.1%}" if known else "- Overall decommit/flip rate: n/a")
    lines.append("")
    lines.append("| Outcome | Count |")
    lines.append("| --- | ---: |")
    for outcome, count in by_outcome.most_common():
        lines.append(f"| {outcome} | {count} |")
    lines.append("")
    lines.append("| Score Bucket | Known Events | Decommit/Flip | Decommit/Flip Rate | Stick Rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in bucket_rows:
        rate = row["decommit_or_flip_rate"]
        stick = row["stick_rate"]
        rate_text = f"{float(rate):.1%}" if rate != "" else ""
        stick_text = f"{float(stick):.1%}" if stick != "" else ""
        lines.append(
            f"| {row['event_strength_bucket']} | {row['commitment_events_with_known_outcome']} | "
            f"{row['decommit_or_flip_events']} | {rate_text} | {stick_text} |"
        )
    lines.append("")
    lines.append("This backtest scores each historical commitment event using pre-outcome signals only:")
    lines.append("timing, prior commitment/decommit history, pre-commit offers/visits, rating tier, distance, and class crowding.")
    lines.append("It does not use post-commit events as an input, because those are the outcome being tested.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-file", type=Path, default=Path("data/commitment_strength_scores.csv"))
    parser.add_argument("--timeline-file", type=Path, default=Path("data/timeline_events.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    recruits = [
        row for row in dedupe_commits(load_csv(args.score_file))
        if in_high_school_cycle(row, "committed_date")
    ]
    events = [row for row in load_csv(args.timeline_file) if in_high_school_cycle(row)]
    recruit_by_key = {(row["class_year"], row["player_id"]): row for row in recruits}
    grouped_events = events_by_player(events)
    position_counts, position_groups = build_position_context(recruits)

    backtest_rows = []
    for key, player_events in grouped_events.items():
        recruit = recruit_by_key.get(key)
        if not recruit:
            continue
        for event in player_events:
            if event["event_type"] != "Commitment":
                continue
            backtest_rows.append(
                score_commitment_event(recruit, event, player_events, position_counts, position_groups)
            )

    bucket_rows = summarize_bucket(backtest_rows)
    position_rows = summarize_by_position(backtest_rows)

    write_csv(args.out_dir / "commitment_strength_backtest_events.csv", backtest_rows, list(backtest_rows[0].keys()))
    write_csv(args.out_dir / "commitment_strength_backtest_buckets.csv", bucket_rows, list(bucket_rows[0].keys()))
    write_csv(args.out_dir / "commitment_strength_backtest_by_position.csv", position_rows, list(position_rows[0].keys()))
    (args.out_dir / "commitment_strength_backtest_summary.md").write_text(
        markdown_summary(backtest_rows, bucket_rows),
        encoding="utf-8",
    )

    print(f"Wrote {len(backtest_rows)} commitment event rows")
    print(f"Wrote bucket summary to {args.out_dir / 'commitment_strength_backtest_buckets.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
