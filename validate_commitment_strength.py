#!/usr/bin/env python3
"""Validate commitment strength buckets across held-out class years."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from score_commitment_strength import load_csv, write_csv


TRAIN_YEARS = {"2022", "2023", "2024"}
TEST_YEARS = {"2025"}
LIVE_YEARS = {"2026"}
BUCKET_ORDER = ["Weak / Attack", "Vulnerable", "Stable / Monitor", "Strong"]


def known_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["outcome"] != "open"]


def is_unstable(row: dict[str, str]) -> bool:
    return row["outcome"] in {"decommitted", "flipped"}


def bucket_summary(rows: list[dict[str, str]], label: str) -> list[dict[str, object]]:
    out = []
    for bucket in BUCKET_ORDER:
        bucket_rows = [row for row in rows if row["event_strength_bucket"] == bucket]
        unstable = sum(is_unstable(row) for row in bucket_rows)
        total = len(bucket_rows)
        out.append(
            {
                "sample": label,
                "bucket": bucket,
                "known_events": total,
                "decommit_or_flip_events": unstable,
                "decommit_or_flip_rate": round(unstable / total, 4) if total else "",
                "stick_rate": round(1 - unstable / total, 4) if total else "",
            }
        )
    return out


def auc(rows: list[dict[str, str]]) -> float | str:
    scored = [
        (float(row["event_vulnerability_score"]), 1 if is_unstable(row) else 0)
        for row in rows
    ]
    positives = sum(label for _, label in scored)
    negatives = len(scored) - positives
    if not positives or not negatives:
        return ""

    ranked = sorted(scored, key=lambda item: item[0])
    rank_sum = 0.0
    index = 1
    while index <= len(ranked):
        start = index
        score = ranked[index - 1][0]
        while index <= len(ranked) and ranked[index - 1][0] == score:
            index += 1
        avg_rank = (start + index - 1) / 2
        rank_sum += sum(label for _, label in ranked[start - 1 : index - 1]) * avg_rank

    return round((rank_sum - positives * (positives + 1) / 2) / (positives * negatives), 4)


def sample_label(year: str) -> str:
    if year in TRAIN_YEARS:
        return "train_2022_2024"
    if year in TEST_YEARS:
        return "holdout_2025"
    if year in LIVE_YEARS:
        return "live_2026"
    return f"class_{year}"


def markdown_summary(rows: list[dict[str, str]], bucket_rows: list[dict[str, object]]) -> str:
    known = known_rows(rows)
    sample_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in known:
        sample_rows[sample_label(row["class_year"])].append(row)

    lines = ["# Commitment Strength Holdout Validation", ""]
    lines.append(f"- Known commitment-event outcomes: {len(known)}")
    lines.append("- Train sample: classes 2022-2024")
    lines.append("- Holdout sample: class 2025")
    lines.append("- Live/current-ish sample: class 2026, interpret carefully because outcomes can still be evolving")
    lines.append("")
    lines.append("| Sample | Known Events | Decommit/Flip Rate | AUC |")
    lines.append("| --- | ---: | ---: | ---: |")
    for label in ["train_2022_2024", "holdout_2025", "live_2026"]:
        sample = sample_rows.get(label, [])
        unstable = sum(is_unstable(row) for row in sample)
        rate = unstable / len(sample) if sample else 0
        auc_value = auc(sample)
        lines.append(
            f"| {label} | {len(sample)} | {rate:.1%} | {auc_value} |"
        )
    lines.append("")
    lines.append("| Sample | Bucket | Known Events | Decommit/Flip Rate | Stick Rate |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in bucket_rows:
        rate = row["decommit_or_flip_rate"]
        stick = row["stick_rate"]
        lines.append(
            f"| {row['sample']} | {row['bucket']} | {row['known_events']} | "
            f"{float(rate):.1%} | {float(stick):.1%} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-file", type=Path, default=Path("data/commitment_strength_backtest_events.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    rows = known_rows(load_csv(args.events_file))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[sample_label(row["class_year"])].append(row)

    bucket_rows: list[dict[str, object]] = []
    for label in ["train_2022_2024", "holdout_2025", "live_2026"]:
        bucket_rows.extend(bucket_summary(grouped[label], label))

    sample_rows = []
    for label in ["train_2022_2024", "holdout_2025", "live_2026"]:
        sample = grouped[label]
        unstable = sum(is_unstable(row) for row in sample)
        sample_rows.append(
            {
                "sample": label,
                "known_events": len(sample),
                "decommit_or_flip_events": unstable,
                "decommit_or_flip_rate": round(unstable / len(sample), 4) if sample else "",
                "auc": auc(sample),
            }
        )

    write_csv(args.out_dir / "commitment_strength_holdout_buckets.csv", bucket_rows, list(bucket_rows[0].keys()))
    write_csv(args.out_dir / "commitment_strength_holdout_samples.csv", sample_rows, list(sample_rows[0].keys()))
    (args.out_dir / "commitment_strength_holdout_summary.md").write_text(
        markdown_summary(rows, bucket_rows),
        encoding="utf-8",
    )
    print("Wrote holdout validation outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
