#!/usr/bin/env python3
"""Train a historical flip/decommit probability model.

This replaces hand-weighted probability language with a trained, time-split
model. Texas-specific targeting remains tags/filters, not model probability.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "days_to_early_signing_day",
    "prior_commitments_at_event",
    "prior_decommits_at_event",
    "offers_before_event",
    "official_visits_before_event",
    "unofficial_visits_before_event",
    "same_position_commits_at_school",
    "higher_rated_same_position_commits",
    "rating",
    "home_to_school_miles",
]

CATEGORICAL_FEATURES = [
    "position",
    "star_bucket",
    "distance_bucket",
    "is_in_state_commit",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TRAIN_YEARS = {2022, 2023, 2024}
HOLDOUT_YEARS = {2025}
LIVE_YEARS = {2026}


def pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else ""


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def prepare_event_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df[df["outcome"].isin(["stuck", "decommitted", "flipped"])].copy()
    df["class_year"] = numeric(df["class_year"]).astype(int)
    df["target"] = df["outcome"].isin(["decommitted", "flipped"]).astype(int)
    for col in NUMERIC_FEATURES:
        df[col] = numeric(df[col])
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("").astype(str)
    return df


def prepare_current_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()
    df["prior_commitments_at_event"] = numeric(df.get("prior_commitments", pd.Series(dtype=str)))
    df["prior_decommits_at_event"] = numeric(df.get("prior_decommits", pd.Series(dtype=str)))
    df["offers_before_event"] = numeric(df.get("offers_before_commit", pd.Series(dtype=str)))
    df["official_visits_before_event"] = numeric(df.get("official_visits_before_commit", pd.Series(dtype=str)))
    df["unofficial_visits_before_event"] = numeric(df.get("unofficial_visits_before_commit", pd.Series(dtype=str)))
    for col in [
        "days_to_early_signing_day",
        "same_position_commits_at_school",
        "higher_rated_same_position_commits",
        "rating",
        "home_to_school_miles",
    ]:
        df[col] = numeric(df.get(col, pd.Series(dtype=str)))
    for col in CATEGORICAL_FEATURES:
        df[col] = df.get(col, pd.Series(dtype=str)).fillna("").astype(str)
    return df


def build_model() -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    base = LogisticRegression(max_iter=3000)
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    return Pipeline(steps=[("preprocess", preprocess), ("model", calibrated)])


def metric_row(label: str, df: pd.DataFrame, probabilities: pd.Series) -> dict[str, object]:
    target = df["target"]
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    row = {
        "sample": label,
        "rows": len(df),
        "decommit_or_flip_events": positives,
        "stick_events": negatives,
        "actual_decommit_flip_rate": round(positives / len(df), 4) if len(df) else "",
        "average_predicted_probability": round(float(probabilities.mean()), 4) if len(df) else "",
    }
    if positives and negatives:
        row["auc"] = round(float(roc_auc_score(target, probabilities)), 4)
    else:
        row["auc"] = ""
    row["brier_score"] = round(float(brier_score_loss(target, probabilities)), 4) if len(df) else ""
    row["log_loss"] = round(float(log_loss(target, probabilities, labels=[0, 1])), 4) if len(df) else ""
    return row


def probability_buckets(df: pd.DataFrame) -> list[dict[str, object]]:
    bucketed = df.copy()
    bucketed["probability_bucket"] = pd.cut(
        bucketed["model_decommit_flip_probability"],
        bins=[0, 0.10, 0.20, 0.30, 0.40, 1.0],
        labels=["0-10%", "10-20%", "20-30%", "30-40%", "40%+"],
        include_lowest=True,
    )
    rows = []
    for bucket, group in bucketed.groupby("probability_bucket", observed=True):
        positives = int(group["target"].sum())
        total = len(group)
        rows.append(
            {
                "probability_bucket": str(bucket),
                "known_events": total,
                "decommit_or_flip_events": positives,
                "actual_decommit_flip_rate": round(positives / total, 4) if total else "",
                "average_predicted_probability": round(float(group["model_decommit_flip_probability"].mean()), 4) if total else "",
            }
        )
    return rows


def rolling_backtest_rows(events: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for test_year in [2023, 2024, 2025, 2026]:
        train_years = list(range(2022, test_year))
        train = events[events["class_year"].isin(train_years)].copy()
        test = events[events["class_year"] == test_year].copy()
        if train.empty or test.empty:
            continue
        model = build_model()
        model.fit(train[FEATURES], train["target"])
        probabilities = pd.Series(model.predict_proba(test[FEATURES])[:, 1], index=test.index)
        row = metric_row(f"train_{train_years[0]}_{train_years[-1]}__test_{test_year}", test, probabilities)
        row["train_years"] = f"{train_years[0]}-{train_years[-1]}"
        row["test_year"] = test_year
        top_20 = test.assign(probability=probabilities).sort_values("probability", ascending=False).head(20)
        row["top_20_decommit_or_flip_events"] = int(top_20["target"].sum())
        row["top_20_hit_rate"] = round(float(top_20["target"].mean()), 4) if len(top_20) else ""
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-file", type=Path, default=Path("data/commitment_strength_backtest_events.csv"))
    parser.add_argument("--scores-file", type=Path, default=Path("data/commitment_strength_scores.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    events = prepare_event_frame(load_csv(args.events_file))
    train = events[events["class_year"].isin(TRAIN_YEARS)].copy()
    holdout = events[events["class_year"].isin(HOLDOUT_YEARS)].copy()
    live = events[events["class_year"].isin(LIVE_YEARS)].copy()

    model = build_model()
    model.fit(train[FEATURES], train["target"])

    event_outputs = events.copy()
    event_outputs["model_decommit_flip_probability"] = model.predict_proba(events[FEATURES])[:, 1]
    event_outputs["model_stick_probability"] = 1 - event_outputs["model_decommit_flip_probability"]

    validation_rows = []
    for label, sample in [("train_2022_2024", train), ("holdout_2025", holdout), ("live_2026", live)]:
        if len(sample):
            probabilities = pd.Series(model.predict_proba(sample[FEATURES])[:, 1], index=sample.index)
            validation_rows.append(metric_row(label, sample, probabilities))

    known_output = event_outputs[event_outputs["class_year"].isin(TRAIN_YEARS | HOLDOUT_YEARS | LIVE_YEARS)].copy()
    bucket_rows = probability_buckets(known_output)
    rolling_rows = rolling_backtest_rows(events)

    score_rows = load_csv(args.scores_file)
    current = prepare_current_frame(score_rows)
    score_probs = model.predict_proba(current[FEATURES])[:, 1]
    for row, probability in zip(score_rows, score_probs, strict=True):
        if not row.get("committed_team") or not row.get("committed_date"):
            row["model_decommit_flip_probability"] = ""
            row["model_stick_probability"] = ""
            continue
        row["model_decommit_flip_probability"] = pct(float(probability))
        row["model_stick_probability"] = pct(float(1 - probability))

    write_csv(args.scores_file, score_rows, list(score_rows[0].keys()))
    write_csv(args.out_dir / "flip_model_validation.csv", validation_rows, list(validation_rows[0].keys()))
    write_csv(args.out_dir / "flip_model_probability_buckets.csv", bucket_rows, list(bucket_rows[0].keys()))
    write_csv(args.out_dir / "flip_model_rolling_backtest.csv", rolling_rows, list(rolling_rows[0].keys()))

    event_export = event_outputs.copy()
    event_export["model_decommit_flip_probability"] = event_export["model_decommit_flip_probability"].map(lambda value: round(float(value), 4))
    event_export["model_stick_probability"] = event_export["model_stick_probability"].map(lambda value: round(float(value), 4))
    write_csv(
        args.out_dir / "flip_model_events.csv",
        event_export.to_dict("records"),
        list(event_export.columns),
    )

    print(f"Trained model on {len(train)} known commitment events")
    print(f"Wrote model probabilities into {args.scores_file}")
    print(f"Wrote validation to {args.out_dir / 'flip_model_validation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
