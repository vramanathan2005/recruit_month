#!/usr/bin/env python3
"""Build staff-facing flip board exports from commitment strength scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from geography_features import coords_for_team, distance_bucket, miles_from_high_school_to_team, team_state
from score_commitment_strength import load_csv, norm_team, write_csv


# Manually maintained "power conference / Notre Dame" helper list. Treat this
# as a useful filter, not an authoritative conference-membership database.
POWER_TEAMS = {
    "alabama", "arkansas", "auburn", "florida", "georgia", "kentucky", "lsu",
    "mississippi state", "missouri", "oklahoma", "ole miss", "south carolina",
    "tennessee", "texas", "texas a&m", "vanderbilt",
    "illinois", "indiana", "iowa", "maryland", "michigan", "michigan state",
    "minnesota", "nebraska", "northwestern", "ohio state", "oregon", "penn state",
    "purdue", "rutgers", "ucla", "usc", "washington", "wisconsin",
    "arizona", "arizona state", "baylor", "byu", "cincinnati", "colorado",
    "houston", "iowa state", "kansas", "kansas state", "oklahoma state",
    "tcu", "texas tech", "ucf", "utah", "west virginia",
    "boston college", "california", "clemson", "duke", "florida state",
    "georgia tech", "louisville", "miami", "nc state", "north carolina",
    "pittsburgh", "smu", "stanford", "syracuse", "virginia", "virginia tech",
    "wake forest",
    "notre dame",
}

TEXAS_REGIONAL_RIVALS = {
    "texas a&m",
    "oklahoma",
    "texas tech",
    "baylor",
    "tcu",
    "houston",
    "smu",
    "lsu",
    "arkansas",
    "oklahoma state",
}

BOARD_FIELDS = [
    "board_rank",
    "class_year",
    "player_id",
    "name",
    "position",
    "committed_team",
    "commitment_type",
    "is_signed_or_enrolled",
    "is_power_team_guess",
    "commitment_strength_score",
    "strength_tier",
    "live_commitment_strength_score",
    "live_strength_tier",
    "model_decommit_flip_probability",
    "model_stick_probability",
    "historical_bucket_decommit_flip_rate",
    "estimated_decommit_flip_probability",
    "estimated_stick_probability",
    "rating",
    "star_bucket",
    "national_rank",
    "committed_date",
    "high_school",
    "home_state",
    "home_city",
    "home_lat",
    "home_lon",
    "college_state",
    "college_lat",
    "college_lon",
    "home_to_school_miles",
    "distance_bucket",
    "geography_precision",
    "is_in_state_commit",
    "days_to_early_signing_day",
    "prior_commitments",
    "prior_decommits",
    "same_position_commits_at_school",
    "higher_rated_same_position_commits",
    "offers_before_commit",
    "official_visits_before_commit",
    "unofficial_visits_before_commit",
    "post_commit_committed_school_visits",
    "post_commit_other_school_offers",
    "post_commit_other_school_visits",
    "post_commit_other_school_official_visits",
    "post_commit_other_team_events",
    "why_flagged",
    "profile_url",
]

BOSS_VIEW_FIELDS = [
    "board_rank",
    "class_year",
    "player_id",
    "name",
    "position",
    "committed_team",
    "commitment_type",
    "is_signed_or_enrolled",
    "is_power_team_guess",
    "live_commitment_strength_score",
    "commitment_strength_score",
    "live_strength_tier",
    "model_decommit_flip_probability",
    "model_stick_probability",
    "estimated_decommit_flip_probability",
    "estimated_stick_probability",
    "historical_bucket_decommit_flip_rate",
    "star_bucket",
    "national_rank",
    "committed_date",
    "high_school",
    "home_state",
    "home_city",
    "home_lat",
    "home_lon",
    "college_state",
    "college_lat",
    "college_lon",
    "home_to_school_miles",
    "distance_bucket",
    "geography_precision",
    "post_commit_other_school_official_visits",
    "post_commit_other_school_visits",
    "post_commit_other_school_offers",
    "post_commit_committed_school_visits",
    "why_flagged",
    "profile_url",
]

TARGET_BOARD_FIELDS = [
    "board_rank",
    "texas_opportunity_label",
    "texas_opportunity_reasons",
    "texas_context_tags",
    "staff_action",
    "data_confidence",
    "data_notes",
    "staff_priority",
    "staff_owner",
    "analyst_status",
    "last_contact_date",
    "next_action",
    "texas_offer",
    "texas_visit",
    "texas_recruiting",
    "do_not_pursue",
    "staff_notes",
    "class_year",
    "player_id",
    "name",
    "position",
    "committed_team",
    "commitment_type",
    "is_signed_or_enrolled",
    "is_power_team_guess",
    "live_commitment_strength_score",
    "commitment_strength_score",
    "live_strength_tier",
    "model_decommit_flip_probability",
    "model_stick_probability",
    "estimated_decommit_flip_probability",
    "estimated_stick_probability",
    "star_bucket",
    "national_rank",
    "committed_date",
    "high_school",
    "home_city",
    "home_state",
    "home_lat",
    "home_lon",
    "target_team",
    "target_state",
    "target_lat",
    "target_lon",
    "home_to_target_miles",
    "target_distance_bucket",
    "geography_precision",
    "committed_school_distance_bucket",
    "post_commit_other_school_official_visits",
    "post_commit_other_school_visits",
    "post_commit_other_school_offers",
    "post_commit_committed_school_visits",
    "why_flagged",
    "profile_url",
]

STAFF_OVERRIDE_FIELDS = [
    "profile_url",
    "staff_priority",
    "staff_owner",
    "analyst_status",
    "last_contact_date",
    "next_action",
    "confidence_override",
    "texas_offer",
    "texas_visit",
    "texas_recruiting",
    "do_not_pursue",
    "staff_notes",
]


def score_float(row: dict[str, str]) -> float:
    try:
        return float(row.get("live_commitment_strength_score") or row["commitment_strength_score"])
    except ValueError:
        return 999.0


def probability_float(value: object) -> float:
    try:
        text = str(value or "").strip()
        if text.endswith("%"):
            return float(text[:-1]) / 100
        return float(text)
    except ValueError:
        return 0.0


def rating_float(row: dict[str, str]) -> float:
    try:
        return float(row["rating"])
    except ValueError:
        return 0.0


def national_rank_int(row: dict[str, str]) -> int:
    try:
        return int(row["national_rank"])
    except ValueError:
        return 999999


def int_value(row: dict[str, object], field: str) -> int:
    try:
        return int(float(str(row.get(field, "") or 0)))
    except ValueError:
        return 0


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "x"}


def load_staff_overrides(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    overrides = {}
    for row in load_csv(path):
        profile_url = str(row.get("profile_url", "")).strip()
        if profile_url:
            overrides[profile_url] = row
    return overrides


def ensure_staff_override_template(path: Path) -> None:
    if path.exists():
        rows = load_csv(path)
        if rows:
            existing_fields = list(rows[0].keys())
        else:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                existing_fields = next(reader, [])
        missing_fields = [field for field in STAFF_OVERRIDE_FIELDS if field not in existing_fields]
        if missing_fields:
            merged_fields = existing_fields + missing_fields
            write_csv(path, [{field: row.get(field, "") for field in merged_fields} for row in rows], merged_fields)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, [], STAFF_OVERRIDE_FIELDS)


def apply_staff_override(row: dict[str, object], override: dict[str, str]) -> None:
    for field in STAFF_OVERRIDE_FIELDS:
        if field == "profile_url":
            continue
        row[field] = override.get(field, "")


def staff_signal_reasons(row: dict[str, object]) -> list[str]:
    reasons = []
    if truthy(row.get("staff_priority")):
        reasons.append("staff priority")
    if truthy(row.get("texas_offer")):
        reasons.append("Texas offer")
    if truthy(row.get("texas_visit")):
        reasons.append("Texas visit")
    if truthy(row.get("texas_recruiting")):
        reasons.append("Texas recruiting")
    return reasons


def data_confidence(row: dict[str, object]) -> tuple[str, str]:
    override = str(row.get("confidence_override", "")).strip()
    if override:
        return override, "staff override"

    notes = []
    precision = str(row.get("geography_precision", ""))
    if precision in {"city", "city_to_state"}:
        geo_score = 2
        notes.append("city-level geography")
    elif precision == "state":
        geo_score = 1
        notes.append("state-level geography")
    else:
        geo_score = 0
        notes.append("missing geography")

    if row.get("committed_date"):
        notes.append("commit date captured")
        date_score = 1
    else:
        notes.append("missing commit date")
        date_score = 0

    activity_total = sum(
        int_value(row, field)
        for field in [
            "post_commit_other_school_official_visits",
            "post_commit_other_school_visits",
            "post_commit_other_school_offers",
            "post_commit_committed_school_visits",
        ]
    )
    if activity_total:
        notes.append("post-commit activity captured")
        activity_score = 1
    else:
        notes.append("no post-commit activity captured")
        activity_score = 0

    total = geo_score + date_score + activity_score
    if total >= 4:
        return "High", "; ".join(notes)
    if total >= 2:
        return "Medium", "; ".join(notes)
    return "Low", "; ".join(notes)


def staff_action(row: dict[str, object], opportunity_label: str, confidence: str) -> str:
    if truthy(row.get("do_not_pursue")):
        return "Do Not Chase"
    if truthy(row.get("staff_priority")):
        return "Call Today"
    if confidence == "Low":
        return "Need Intel"
    if opportunity_label == "Attack Now":
        return "Call Today"
    if opportunity_label == "Strong Texas Angle":
        return "Coach Review"
    if opportunity_label == "Need Texas Signal":
        return "Need Intel"
    risk = probability_float(row.get("model_decommit_flip_probability"))
    if risk >= 0.50 and str(row.get("star_bucket", "")) in {"4-star", "5-star"}:
        return "Coach Review"
    return "Watch"


def is_power_team(team: str) -> bool:
    return norm_team(team) in POWER_TEAMS


def is_signed_or_enrolled(row: dict[str, str]) -> bool:
    return row.get("commitment_type") in {"Signed", "Enrolled"}


def why_flagged(row: dict[str, str]) -> str:
    reasons = []
    days = row.get("days_to_early_signing_day")
    if days:
        try:
            days_int = int(days)
            if days_int > 365:
                reasons.append("very early commit")
            elif days_int > 180:
                reasons.append("early commit")
        except ValueError:
            pass
    if row.get("prior_commitments") == "0":
        reasons.append("first commitment")
    if row.get("prior_decommits") not in {"", "0"}:
        reasons.append("prior decommit")
    if row.get("same_position_commits_at_school") == "0":
        reasons.append("only commit at position")
    elif row.get("same_position_commits_at_school") == "1":
        reasons.append("thin position group")
    if row.get("distance_bucket") in {"far", "cross_country"}:
        reasons.append(row["distance_bucket"].replace("_", " "))
    if row.get("star_bucket") in {"4-star", "5-star"}:
        reasons.append(row["star_bucket"])
    if row.get("post_commit_other_team_events") not in {"", "0"}:
        reasons.append("post-commit other-school activity")
    if row.get("post_commit_other_school_official_visits") not in {"", "0"}:
        reasons.append("other-school OV after commit")
    elif row.get("post_commit_other_school_visits") not in {"", "0"}:
        reasons.append("other-school visit after commit")
    if row.get("post_commit_other_school_offers") not in {"", "0"}:
        reasons.append("other-school offers after commit")
    return "; ".join(reasons[:5])


def percent_text(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else ""


def load_bucket_rates(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rates = {}
    for row in load_csv(path):
        try:
            rates[row["event_strength_bucket"]] = float(row["decommit_or_flip_rate"])
        except (KeyError, ValueError):
            continue
    return rates


def board_rows(rows: list[dict[str, str]], bucket_rates: dict[str, float]) -> list[dict[str, object]]:
    eligible = [row for row in rows if row.get("committed_team") and row.get("committed_date")]
    eligible.sort(
        key=lambda row: (
            -probability_float(row.get("model_decommit_flip_probability")),
            score_float(row),
            -rating_float(row),
            national_rank_int(row),
            row["name"],
        )
    )
    out = []
    for idx, row in enumerate(eligible, 1):
        live_tier = row.get("live_strength_tier", row["strength_tier"])
        base_tier = row["strength_tier"]
        estimated_flip_rate = bucket_rates.get(live_tier)
        base_flip_rate = bucket_rates.get(base_tier)
        out.append(
            {
                "board_rank": idx,
                "class_year": row["class_year"],
                "player_id": row.get("player_id", ""),
                "name": row["name"],
                "position": row["position"],
                "committed_team": row["committed_team"],
                "commitment_type": row["commitment_type"],
                "is_signed_or_enrolled": "yes" if is_signed_or_enrolled(row) else "no",
                "is_power_team_guess": "yes" if is_power_team(row["committed_team"]) else "no",
                "commitment_strength_score": row["commitment_strength_score"],
                "strength_tier": row["strength_tier"],
                "live_commitment_strength_score": row.get("live_commitment_strength_score", row["commitment_strength_score"]),
                "live_strength_tier": live_tier,
                "model_decommit_flip_probability": row.get("model_decommit_flip_probability", ""),
                "model_stick_probability": row.get("model_stick_probability", ""),
                "historical_bucket_decommit_flip_rate": percent_text(base_flip_rate),
                "estimated_decommit_flip_probability": percent_text(estimated_flip_rate),
                "estimated_stick_probability": percent_text(1 - estimated_flip_rate if estimated_flip_rate is not None else None),
                "rating": row["rating"],
                "star_bucket": row["star_bucket"],
                "national_rank": row["national_rank"],
                "committed_date": row["committed_date"],
                "high_school": row.get("high_school", ""),
                "home_city": row.get("home_city", ""),
                "home_state": row.get("home_state", ""),
                "home_lat": row.get("home_lat", ""),
                "home_lon": row.get("home_lon", ""),
                "college_state": row.get("college_state", ""),
                "college_lat": row.get("college_lat", ""),
                "college_lon": row.get("college_lon", ""),
                "home_to_school_miles": row.get("home_to_school_miles", ""),
                "distance_bucket": row.get("distance_bucket", ""),
                "geography_precision": row.get("geography_precision", ""),
                "is_in_state_commit": row.get("is_in_state_commit", ""),
                "days_to_early_signing_day": row["days_to_early_signing_day"],
                "prior_commitments": row["prior_commitments"],
                "prior_decommits": row["prior_decommits"],
                "same_position_commits_at_school": row["same_position_commits_at_school"],
                "higher_rated_same_position_commits": row["higher_rated_same_position_commits"],
                "offers_before_commit": row.get("offers_before_commit", ""),
                "official_visits_before_commit": row.get("official_visits_before_commit", ""),
                "unofficial_visits_before_commit": row.get("unofficial_visits_before_commit", ""),
                "post_commit_committed_school_visits": row.get("post_commit_committed_school_visits", ""),
                "post_commit_other_school_offers": row.get("post_commit_other_school_offers", ""),
                "post_commit_other_school_visits": row.get("post_commit_other_school_visits", ""),
                "post_commit_other_school_official_visits": row.get("post_commit_other_school_official_visits", ""),
                "post_commit_other_team_events": row["post_commit_other_team_events"],
                "why_flagged": why_flagged(row),
                "profile_url": row["profile_url"],
            }
        )
    return out


def boss_view_rows(rows: list[dict[str, object]], limit: int | None = None) -> list[dict[str, object]]:
    selected = rows[:limit] if limit else rows
    return [{field: row.get(field, "") for field in BOSS_VIEW_FIELDS} for row in selected]


def target_board_rows(
    rows: list[dict[str, object]],
    target_team: str,
    class_year: str,
    staff_overrides: dict[str, dict[str, str]] | None = None,
    p4_only: bool = True,
    exclude_signed: bool = True,
) -> list[dict[str, object]]:
    target_key = norm_team(target_team)
    target_state = team_state(target_key)
    target_coords = coords_for_team(target_key)
    staff_overrides = staff_overrides or {}
    eligible = []
    for row in rows:
        if row["class_year"] != class_year:
            continue
        if p4_only and row["is_power_team_guess"] != "yes":
            continue
        if exclude_signed and row["is_signed_or_enrolled"] == "yes":
            continue
        if norm_team(str(row["committed_team"])) == target_key:
            continue

        copied = row.copy()
        override = staff_overrides.get(str(copied.get("profile_url", "")).strip(), {})
        apply_staff_override(copied, override)
        if truthy(copied.get("do_not_pursue")):
            continue
        miles, precision = miles_from_high_school_to_team(str(copied.get("high_school", "")), target_team)
        copied["target_team"] = target_team
        copied["target_state"] = target_state
        copied["target_lat"] = round(target_coords[0], 6) if target_coords else ""
        copied["target_lon"] = round(target_coords[1], 6) if target_coords else ""
        copied["home_to_target_miles"] = miles if miles is not None else ""
        copied["target_distance_bucket"] = distance_bucket(miles)
        copied["geography_precision"] = precision or copied.get("geography_precision", "")
        copied["committed_school_distance_bucket"] = copied.get("distance_bucket", "")
        tags = texas_context_tags(row=copied)
        staff_reasons = staff_signal_reasons(copied)
        tags.extend(staff_reasons)
        copied["texas_context_tags"] = "; ".join(tags[:6])
        opportunity_label, opportunity_reasons = texas_opportunity(row=copied, tags=tags)
        if staff_reasons:
            opportunity_reasons = staff_reasons + [reason for reason in opportunity_reasons if reason not in staff_reasons]
            if truthy(copied.get("staff_priority")):
                opportunity_label = "Attack Now"
            elif opportunity_label in {"Monitor", "Need Texas Signal"}:
                opportunity_label = "Strong Texas Angle"
        copied["texas_opportunity_label"] = opportunity_label
        copied["texas_opportunity_reasons"] = "; ".join(opportunity_reasons)
        confidence, confidence_notes = data_confidence(copied)
        copied["data_confidence"] = confidence
        copied["data_notes"] = confidence_notes
        copied["staff_action"] = staff_action(copied, opportunity_label, confidence)
        eligible.append(copied)

    opportunity_rank = {
        "Attack Now": 0,
        "Strong Texas Angle": 1,
        "Monitor": 2,
        "Need Texas Signal": 3,
    }
    action_rank = {
        "Call Today": 0,
        "Coach Review": 1,
        "Need Intel": 2,
        "Watch": 3,
        "Do Not Chase": 4,
    }
    eligible.sort(
        key=lambda row: (
            0 if truthy(row.get("staff_priority")) else 1,
            action_rank.get(str(row.get("staff_action", "")), 9),
            opportunity_rank.get(str(row.get("texas_opportunity_label", "")), 9),
            -probability_float(row.get("model_decommit_flip_probability")),
            score_float(row),
            row["home_to_target_miles"] if isinstance(row["home_to_target_miles"], int) else 999999,
            -rating_float(row),
            national_rank_int(row),
            row["name"],
        )
    )
    out = []
    for idx, row in enumerate(eligible, 1):
        copied = {field: row.get(field, "") for field in TARGET_BOARD_FIELDS}
        copied["board_rank"] = idx
        out.append(copied)
    return out


def texas_opportunity(row: dict[str, object], tags: list[str]) -> tuple[str, list[str]]:
    reasons = []
    has_geography = any(tag in tags for tag in ["Texas recruit", "nearby"])
    has_regional_proximity = "regional" in tags
    has_rival = "regional/rival commit" in tags
    has_blue_chip = str(row.get("star_bucket", "")) in {"4-star", "5-star"}
    has_activity = any(
        int_value(row, field) > 0
        for field in [
            "post_commit_other_school_official_visits",
            "post_commit_other_school_visits",
            "post_commit_other_school_offers",
        ]
    )
    if has_geography:
        reasons.append("geographic fit")
    elif has_regional_proximity:
        reasons.append("regional proximity")
    if has_rival:
        reasons.append("regional/rival school")
    if has_blue_chip:
        reasons.append(str(row.get("star_bucket", "")))
    if has_activity:
        reasons.append("active after commitment")

    has_texas_angle = has_geography or has_rival

    if has_activity and has_texas_angle:
        return "Attack Now", reasons
    if (has_geography or has_rival) and has_blue_chip:
        return "Strong Texas Angle", reasons
    if has_texas_angle:
        return "Monitor", reasons
    if has_regional_proximity:
        return "Monitor", reasons
    if has_activity:
        if has_blue_chip:
            return "Need Texas Signal", [str(row.get("star_bucket", "")), "active after commitment", "no clear Texas-specific signal captured"]
        return "Need Texas Signal", ["active after commitment", "no clear Texas-specific signal captured"]
    if has_blue_chip:
        return "Monitor", reasons
    return "Need Texas Signal", ["no clear Texas-specific signal captured"]


def texas_context_tags(row: dict[str, object]) -> list[str]:
    tags = []
    bucket = str(row.get("target_distance_bucket", ""))
    if bucket == "in_state":
        tags.append("Texas recruit")
    elif bucket == "nearby":
        tags.append("nearby")
    elif bucket == "regional":
        tags.append("regional")

    star_bucket = str(row.get("star_bucket", ""))
    if star_bucket in {"4-star", "5-star"}:
        tags.append(star_bucket)

    if norm_team(str(row.get("committed_team", ""))) in TEXAS_REGIONAL_RIVALS:
        tags.append("regional/rival commit")

    other_school_ovs = int_value(row, "post_commit_other_school_official_visits")
    other_school_visits = int_value(row, "post_commit_other_school_visits")
    other_school_offers = int_value(row, "post_commit_other_school_offers")
    committed_school_visits = int_value(row, "post_commit_committed_school_visits")

    if other_school_ovs:
        tags.append("other-school OV after commit")
    elif other_school_visits:
        tags.append("other-school visit after commit")

    if other_school_offers:
        tags.append("other-school offers after commit")

    if committed_school_visits:
        tags.append("visited committed school")

    return tags


def rerank(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    reranked = []
    for idx, row in enumerate(rows, 1):
        copied = row.copy()
        copied["board_rank"] = idx
        reranked.append(copied)
    return reranked


def write_position_boards(out_dir: Path, rows: list[dict[str, object]], class_year: str | None = None) -> list[dict[str, object]]:
    summary = []
    by_position: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if class_year and row["class_year"] != class_year:
            continue
        by_position.setdefault(str(row["position"] or "UNK"), []).append(row)

    for position, position_rows in sorted(by_position.items()):
        top = rerank(position_rows[:25])
        path = out_dir / f"weakest_{class_year + '_' if class_year else ''}{position.lower()}_commits.csv"
        write_csv(path, top, BOARD_FIELDS)
        summary.append({"class_year": class_year or "all", "position": position, "rows": len(top), "file": str(path)})
    return summary


WEEKLY_BRIEF_FIELDS = [
    "board_rank",
    "staff_action",
    "data_confidence",
    "name",
    "position",
    "committed_team",
    "star_bucket",
    "national_rank",
    "model_decommit_flip_probability",
    "model_stick_probability",
    "home_city",
    "home_state",
    "home_to_target_miles",
    "texas_opportunity_label",
    "texas_opportunity_reasons",
    "why_flagged",
    "staff_owner",
    "analyst_status",
    "last_contact_date",
    "next_action",
    "staff_notes",
    "profile_url",
]


def write_weekly_brief(out_dir: Path, target_rows: list[dict[str, object]], target_slug: str, class_year: str) -> None:
    selected = [row for row in target_rows if row.get("staff_action") in {"Call Today", "Coach Review", "Need Intel"}]
    brief_rows = [{field: row.get(field, "") for field in WEEKLY_BRIEF_FIELDS} for row in selected[:50]]
    write_csv(
        out_dir / f"weekly_brief_{target_slug}_{class_year}.csv",
        brief_rows,
        WEEKLY_BRIEF_FIELDS,
    )


def markdown_summary(all_rows: list[dict[str, object]], current_rows: list[dict[str, object]], p4_rows: list[dict[str, object]]) -> str:
    lines = ["# Flip Board Exports", ""]
    lines.append(f"- All committed recruits on board: {len(all_rows)}")
    current_class = current_rows[0]["class_year"] if current_rows else "current"
    lines.append(f"- {current_class} committed recruits on board: {len(current_rows)}")
    lines.append(f"- Power-team guess rows: {len(p4_rows)}")
    lines.append("")
    lines.append("| Rank | Player | Class | Pos | Team | Model Flip | Live Score | Base Score | Bucket | Risk Signals |")
    lines.append("| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |")
    for row in current_rows[:25]:
        lines.append(
            f"| {row['board_rank']} | {row['name']} | {row['class_year']} | {row['position']} | "
            f"{row['committed_team']} | {row.get('model_decommit_flip_probability', '')} | "
            f"{row['live_commitment_strength_score']} | "
            f"{row['commitment_strength_score']} | {row['live_strength_tier']} | "
            f"{row['why_flagged']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-file", type=Path, default=Path("data/commitment_strength_scores.csv"))
    parser.add_argument("--bucket-file", type=Path, default=Path("data/commitment_strength_backtest_buckets.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/flip_boards"))
    parser.add_argument("--current-class", default="2027")
    parser.add_argument("--target-team", default="Texas")
    parser.add_argument("--staff-overrides-file", type=Path, default=Path("data/texas_staff_overrides.csv"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ensure_staff_override_template(args.staff_overrides_file)
    staff_overrides = load_staff_overrides(args.staff_overrides_file)
    rows = board_rows(load_csv(args.scores_file), load_bucket_rates(args.bucket_file))
    current_rows = rerank([row for row in rows if row["class_year"] == args.current_class])
    p4_rows = rerank([row for row in rows if row["is_power_team_guess"] == "yes"])
    current_p4_rows = rerank([row for row in current_rows if row["is_power_team_guess"] == "yes"])
    target_rows = target_board_rows(rows, args.target_team, args.current_class, staff_overrides)

    write_csv(args.out_dir / "flip_board_all_commits.csv", rows, BOARD_FIELDS)
    write_csv(args.out_dir / f"weakest_{args.current_class}_commits.csv", current_rows[:250], BOARD_FIELDS)
    write_csv(args.out_dir / "weakest_power_team_commits.csv", p4_rows[:500], BOARD_FIELDS)
    write_csv(args.out_dir / f"weakest_{args.current_class}_power_team_commits.csv", current_p4_rows[:250], BOARD_FIELDS)
    write_csv(args.out_dir / "boss_view_all_commits.csv", boss_view_rows(rows), BOSS_VIEW_FIELDS)
    write_csv(args.out_dir / f"boss_view_{args.current_class}_commits.csv", boss_view_rows(current_rows, 250), BOSS_VIEW_FIELDS)
    write_csv(args.out_dir / "boss_view_power_team_commits.csv", boss_view_rows(p4_rows, 500), BOSS_VIEW_FIELDS)
    write_csv(args.out_dir / f"boss_view_{args.current_class}_power_team_commits.csv", boss_view_rows(current_p4_rows, 250), BOSS_VIEW_FIELDS)
    target_slug = norm_team(args.target_team).replace(" ", "_")
    write_csv(args.out_dir / f"target_board_{target_slug}_{args.current_class}.csv", target_rows[:250], TARGET_BOARD_FIELDS)
    write_weekly_brief(args.out_dir, target_rows, target_slug, args.current_class)
    position_summary = write_position_boards(args.out_dir, rows)
    position_summary.extend(write_position_boards(args.out_dir, rows, args.current_class))
    write_csv(args.out_dir / "position_board_index.csv", position_summary, ["class_year", "position", "rows", "file"])
    (args.out_dir / "flip_board_summary.md").write_text(
        markdown_summary(rows, current_rows, p4_rows),
        encoding="utf-8",
    )
    print(f"Wrote flip boards to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
