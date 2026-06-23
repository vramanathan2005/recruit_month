#!/usr/bin/env python3
"""Create a first-pass commitment strength score from the 247 scrape outputs.

This is a rules-based V1. It intentionally avoids using final outcome fields
like "Signed" or "Enrolled" as positive evidence, because those would leak the
answer for historical classes. The point is to create a transparent vulnerability
ranking we can later backtest and calibrate into probabilities.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

from geography_features import geography_features


SIGNING_MONTH = 12


def parse_date(value: str) -> date | None:
    if not value or value.upper() == "NA":
        return None
    try:
        return datetime.strptime(value, "%m/%d/%Y").date()
    except ValueError:
        return None


def early_signing_day(class_year: int) -> date:
    """Approximate early signing day as the third Wednesday of December."""
    december_first = date(class_year - 1, SIGNING_MONTH, 1)
    days_until_wednesday = (2 - december_first.weekday()) % 7
    first_wednesday = december_first.toordinal() + days_until_wednesday
    return date.fromordinal(first_wednesday + 14)


def high_school_cycle_cutoff(class_year: int) -> date:
    """Last date to treat as part of the player's high-school recruiting cycle."""
    return date(class_year, 8, 1)


def in_high_school_cycle(row: dict[str, str], date_field: str = "event_date") -> bool:
    event_date = parse_date(row.get(date_field, ""))
    if not event_date:
        return True
    try:
        class_year = int(row["class_year"])
    except (KeyError, TypeError, ValueError):
        return True
    return event_date <= high_school_cycle_cutoff(class_year)


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def norm_team(value: str) -> str:
    value = " ".join((value or "").lower().replace("&amp;", "&").split())
    suffixes = (
        " tigers",
        " wildcats",
        " bulldogs",
        " bears",
        " bruins",
        " ducks",
        " longhorns",
        " trojans",
        " crimson tide",
        " buckeyes",
        " wolverines",
        " nittany lions",
        " fighting irish",
        " seminoles",
        " hurricanes",
        " gators",
        " volunteers",
        " rebels",
        " sooners",
        " cowboys",
        " aggies",
        " razorbacks",
        " gamecocks",
        " commodores",
        " yellow jackets",
        " tar heels",
        " wolfpack",
        " blue devils",
        " cavaliers",
        " hokies",
        " mountaineers",
        " cardinals",
        " panthers",
        " huskies",
        " cougars",
        " beavers",
        " utes",
        " sun devils",
        " golden bears",
        " red raiders",
        " horned frogs",
        " cyclones",
        " jayhawks",
        " knights",
        " bearcats",
        " boilermakers",
        " badgers",
        " hawkeyes",
        " spartans",
        " gophers",
        " terrapins",
        " scarlet knights",
        " hoosiers",
        " fighting illini",
    )
    for suffix in suffixes:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dedupe_commits(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["class_year"], row["player_id"])
        current = deduped.get(key)
        if current is None:
            deduped[key] = row
            continue
        current_has_date = bool(current.get("committed_date"))
        row_has_date = bool(row.get("committed_date"))
        if row_has_date and not current_has_date:
            deduped[key] = row
    return list(deduped.values())


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rating_float(row: dict[str, str]) -> float:
    try:
        return float(row.get("rating", ""))
    except ValueError:
        return 0.0


def build_event_index(events: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    by_player: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
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
        by_player[(event["class_year"], event["player_id"])].append(event)
    for key in by_player:
        by_player[key].sort(key=lambda row: parse_date(row["event_date"]) or date.min)
    return by_player


def build_position_context(commits: list[dict[str, str]]) -> tuple[Counter, dict[tuple[str, str, str], list[dict[str, str]]]]:
    counts: Counter = Counter()
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in commits:
        if not row.get("committed_team") or not row.get("position"):
            continue
        key = (row["class_year"], norm_team(row["committed_team"]), row["position"])
        counts[key] += 1
        groups[key].append(row)
    return counts, groups


def strength_tier(strength: float) -> str:
    if strength >= 80:
        return "Strong"
    if strength >= 65:
        return "Stable / Monitor"
    if strength >= 50:
        return "Vulnerable"
    return "Weak / Attack"


def score_row(
    row: dict[str, str],
    events: list[dict[str, str]],
    position_counts: Counter,
    position_groups: dict[tuple[str, str, str], list[dict[str, str]]],
) -> dict[str, object]:
    committed_date = parse_date(row.get("committed_date", ""))
    class_year = int(row["class_year"])
    final_team = norm_team(row.get("committed_team", ""))

    prior_commitments = 0
    prior_decommits = 0
    post_commit_other_team_events = 0
    post_commit_decommits = 0
    offers_before_commit = 0
    offers_after_commit = 0
    official_visits_before_commit = 0
    unofficial_visits_before_commit = 0
    post_commit_committed_school_visits = 0
    post_commit_other_school_offers = 0
    post_commit_other_school_visits = 0
    post_commit_other_school_official_visits = 0

    for event in events:
        event_date = parse_date(event["event_date"])
        if not event_date or not committed_date:
            continue
        event_type = event["event_type"]
        event_team = norm_team(event.get("school", ""))
        if event_date < committed_date:
            prior_commitments += event_type == "Commitment"
            prior_decommits += event_type == "Decommit"
            offers_before_commit += event_type == "Offer"
            official_visits_before_commit += event_type == "Official Visit"
            unofficial_visits_before_commit += event_type == "Unofficial Visit"
        elif event_date > committed_date:
            if event_type == "Decommit":
                post_commit_decommits += 1
            if event_type in {"Commitment", "Decommit"} and event_team and event_team != final_team:
                post_commit_other_team_events += 1
            offers_after_commit += event_type == "Offer"
            if event_type == "Offer" and event_team and event_team != final_team:
                post_commit_other_school_offers += 1
            if event_type in {"Official Visit", "Unofficial Visit"}:
                if event_team and event_team == final_team:
                    post_commit_committed_school_visits += 1
                elif event_team:
                    post_commit_other_school_visits += 1
                    post_commit_other_school_official_visits += event_type == "Official Visit"

    signing_day = early_signing_day(class_year)
    days_to_signing = (signing_day - committed_date).days if committed_date else 0
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

    pos_key = (row["class_year"], final_team, row.get("position", ""))
    same_position_commits = max(position_counts.get(pos_key, 0) - 1, 0)
    current_rating = rating_float(row)
    higher_rated_same_position = 0
    for teammate in position_groups.get(pos_key, []):
        if teammate["player_id"] != row["player_id"] and rating_float(teammate) > current_rating:
            higher_rated_same_position += 1

    history_risk = 10 if prior_commitments == 0 else -5
    if prior_decommits:
        history_risk -= 5

    process_risk = 0
    if offers_before_commit == 0:
        process_risk += 2
    elif offers_before_commit >= 16:
        process_risk -= 4
    elif offers_before_commit >= 6:
        process_risk -= 1
    if official_visits_before_commit:
        process_risk -= 4
    elif unofficial_visits_before_commit:
        process_risk += 2

    post_commit_risk = clamp(post_commit_other_team_events * 8 + post_commit_decommits * 12, 0, 28)
    live_activity_risk = clamp(
        post_commit_risk
        + post_commit_other_school_offers * 0.75
        + post_commit_other_school_visits * 3
        + post_commit_other_school_official_visits * 3
        - post_commit_committed_school_visits * 2,
        0,
        20,
    )
    if same_position_commits == 0:
        crowding_risk = 10
    elif same_position_commits == 1:
        crowding_risk = 5
    elif same_position_commits >= 3:
        crowding_risk = -3
    else:
        crowding_risk = 0

    geo = geography_features(row.get("high_school", ""), final_team)
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

    star_bucket = row.get("star_bucket", "")
    attention_risk = 0
    if star_bucket == "5-star":
        attention_risk = 7
    elif star_bucket == "4-star":
        attention_risk = 4

    missing_data_risk = 8 if not committed_date else 0
    total_risk = clamp(
        early_commit_risk
        + history_risk
        + process_risk
        + crowding_risk
        + distance_risk
        + attention_risk
        + missing_data_risk,
        0,
        100,
    )
    strength = round(100 - total_risk, 1)
    live_total_risk = clamp(total_risk + live_activity_risk, 0, 100)
    live_strength = round(100 - live_total_risk, 1)

    return {
        **row,
        "commitment_strength_score": strength,
        "vulnerability_score": round(total_risk, 1),
        "strength_tier": strength_tier(strength),
        "live_commitment_strength_score": live_strength,
        "live_vulnerability_score": round(live_total_risk, 1),
        "live_strength_tier": strength_tier(live_strength),
        "days_to_early_signing_day": days_to_signing if committed_date else "",
        "prior_commitments": prior_commitments,
        "prior_decommits": prior_decommits,
        "offers_before_commit": offers_before_commit,
        "offers_after_commit": offers_after_commit,
        "official_visits_before_commit": official_visits_before_commit,
        "unofficial_visits_before_commit": unofficial_visits_before_commit,
        "post_commit_committed_school_visits": post_commit_committed_school_visits,
        "post_commit_other_school_offers": post_commit_other_school_offers,
        "post_commit_other_school_visits": post_commit_other_school_visits,
        "post_commit_other_school_official_visits": post_commit_other_school_official_visits,
        "post_commit_other_team_events": post_commit_other_team_events,
        "post_commit_decommits": post_commit_decommits,
        "same_position_commits_at_school": same_position_commits,
        "higher_rated_same_position_commits": higher_rated_same_position,
        "early_commit_risk": round(early_commit_risk, 1),
        "history_risk": round(history_risk, 1),
        "process_risk": round(process_risk, 1),
        "post_commit_risk": round(post_commit_risk, 1),
        "live_activity_risk": round(live_activity_risk, 1),
        "class_crowding_risk": round(crowding_risk, 1),
        **geo,
        "distance_risk": distance_risk,
        "attention_risk": attention_risk,
        "missing_data_risk": missing_data_risk,
    }


def summarize(rows: list[dict[str, object]]) -> str:
    tiers = Counter(str(row["strength_tier"]) for row in rows)
    matched = [row for row in rows if row.get("committed_date")]
    by_year = Counter(str(row["class_year"]) for row in rows)

    lines = ["# Commitment Strength V1", ""]
    lines.append(f"- Scored recruits: {len(rows)}")
    lines.append(f"- With 247 commitment date: {len(matched)}")
    lines.append("- Score meaning: `100` is strongest, `0` is weakest / most vulnerable.")
    lines.append("")
    lines.append("| Tier | Count |")
    lines.append("| --- | ---: |")
    for tier in ["Strong", "Stable / Monitor", "Vulnerable", "Weak / Attack"]:
        lines.append(f"| {tier} | {tiers[tier]} |")
    lines.append("")
    lines.append("| Class | Recruits |")
    lines.append("| --- | ---: |")
    for year in sorted(by_year):
        lines.append(f"| {year} | {by_year[year]} |")
    lines.append("")
    lines.append("## Lowest 20 Scores")
    lines.append("")
    lines.append("| Player | Class | Pos | Team | Score | Why |")
    lines.append("| --- | ---: | --- | --- | ---: | --- |")
    weakest = sorted(rows, key=lambda row: float(row["commitment_strength_score"]))[:20]
    for row in weakest:
        why = (
            f"history {row['history_risk']}, process {row['process_risk']}, post {row['post_commit_risk']}, "
            f"crowding {row['class_crowding_risk']}, time {row['early_commit_risk']}"
        )
        lines.append(
            f"| {row['name']} | {row['class_year']} | {row['position']} | "
            f"{row['committed_team']} | {row['commitment_strength_score']} | {why} |"
        )
    lines.append("")
    lines.append("Base score uses pre-outcome signals only. Live score separately adjusts for post-commit offers/visits.")
    lines.append("Coach stability and social activity still need enrichment tables before they can be scored.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-file", type=Path, default=Path("data/commit_dates.csv"))
    parser.add_argument("--timeline-file", type=Path, default=Path("data/timeline_events.csv"))
    parser.add_argument("--out-file", type=Path, default=Path("data/commitment_strength_scores.csv"))
    parser.add_argument("--summary-file", type=Path, default=Path("data/commitment_strength_summary.md"))
    args = parser.parse_args()

    commits = [
        row for row in dedupe_commits(load_csv(args.commit_file))
        if in_high_school_cycle(row, "committed_date")
    ]
    events = [row for row in load_csv(args.timeline_file) if in_high_school_cycle(row)]
    events_by_player = build_event_index(events)
    position_counts, position_groups = build_position_context(commits)

    scored = [
        score_row(
            row,
            events_by_player.get((row["class_year"], row["player_id"]), []),
            position_counts,
            position_groups,
        )
        for row in commits
    ]

    fieldnames = list(scored[0].keys()) if scored else []
    write_csv(args.out_file, scored, fieldnames)
    args.summary_file.write_text(summarize(scored), encoding="utf-8")
    print(f"Wrote {len(scored)} rows to {args.out_file}")
    print(f"Wrote summary to {args.summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
