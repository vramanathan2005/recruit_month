#!/usr/bin/env python3
"""Bayesian layer on top of the tree model, for the one strong signal the tree can't learn from a handful of
cases: an On3 insider switching his pick away from the player's school.

    final odds = tree's final odds x (switch multiplier, only when an insider has switched away)

Players with no insider switch keep the tree's number exactly. The multiplier is learned as a range:
  - a fresh switch (under 45 days old) and an old one (45+ days, like Easton Royal's) each get a multiplier;
  - both start skeptical ("probably x1, but x1/20 to x20 plausible");
  - the old one borrows from the fresh one — it starts centered on it and the few old-switch cases move it
    ("partial pooling": like judging a kicker with 3 attempts by starting from what kickers usually make).

    .venv/bin/python bayes_layer.py           fit on 2024-26, score live commitments → data/cache/bayes_live.csv
    .venv/bin/python bayes_layer.py --check   leave one class out: learn from two, test on the third

Needs data/cache/before_signing_{held_out,live}.pkl from train_flip_snapshot_model.py.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import train_flip_snapshot_model as m

FRESH_DAYS = 45
CLASSES = [2024, 2025, 2026]
PICKS = Path.home() / "Tars/reports/web_report/backfill"


# ---- the data -------------------------------------------------------------------------------------------
def switch_age(row) -> float:
    """Days since the first insider switched his pick away from the player's school, as of this snapshot."""
    school = m.norm_team(row.committed_school)
    day = date.fromisoformat(row.snapshot_date)
    dates = [date.fromisoformat(p["date"]) for p in m.ON3_PICKS.get(str(row.player_id), [])
             if p["date"][:4].isdigit() and date.fromisoformat(p["date"]) <= day and m.switched_away(p, school)]
    return float((day - min(dates)).days) if dates else np.nan


def signals(frame: pd.DataFrame) -> np.ndarray:
    """[fresh switch, old switch] as 0/1 columns."""
    age = frame["switch_age"].to_numpy()
    switched = ~np.isnan(age)
    return np.column_stack([(switched & (age < FRESH_DAYS)), (switched & (age >= FRESH_DAYS))]).astype(float)


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0.005, 0.995)
    return np.log(p / (1 - p))


# ---- priors ---------------------------------------------------------------------------------------------
def log_prior(theta: np.ndarray) -> float:
    fresh, old = theta
    return -0.5 * (fresh / 1.5) ** 2 - 0.5 * ((old - fresh) / 0.75) ** 2


# ---- fit ------------------------------------------------------------------------------------------------
def sample(X, y, w, offset, draws=6000, burn=3000, chains=3, seed=7) -> np.ndarray:
    """Try many multiplier values; keep them in proportion to how well they explain what happened (x prior)."""
    def score(theta):
        z = offset + X @ theta
        return float(np.sum(w * (y * z - np.logaddexp(0, z)))) + log_prior(theta)
    rng = np.random.default_rng(seed)
    kept = []
    for _ in range(chains):
        theta = rng.normal(0, 0.1, 2)
        current, step, accepted = score(theta), np.full(2, 0.15), 0
        for i in range(draws + burn):
            proposal = theta + rng.normal(0, step)
            new = score(proposal)
            if np.log(rng.random()) < new - current:
                theta, current = proposal, new
                accepted += 1
            if i < burn and i % 200 == 199:
                step *= 1.3 if accepted / 200 > 0.4 else (0.7 if accepted / 200 < 0.2 else 1.0)
                accepted = 0
            if i >= burn:
                kept.append(theta.copy())
    return np.array(kept)


def fit(rows: pd.DataFrame, tree: np.ndarray, **kw) -> np.ndarray:
    # Each snapshot gets the same small weight so rows add up to the number of commitments (honest ranges).
    w = np.full(len(rows), rows[["player_id", "commitment_date"]].drop_duplicates().shape[0] / len(rows))
    return sample(signals(rows), rows["target_before_signing"].to_numpy().astype(float), w, logit(tree), **kw)


def predict(rows: pd.DataFrame, tree: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Every row x every kept draw → probabilities."""
    return 1 / (1 + np.exp(-(logit(tree)[:, None] + signals(rows) @ draws.T)))


def multiplier(values: np.ndarray) -> str:
    lo, mid, hi = np.percentile(np.exp(values), [10, 50, 90])
    return f"x{mid:.2f} (80% range x{lo:.2f} to x{hi:.2f})"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    m.load_on3_picks(PICKS)
    held = pd.read_pickle("data/cache/before_signing_held_out.pkl")
    live = pd.read_pickle("data/cache/before_signing_live.pkl")
    held = held[held["class_year"].isin(CLASSES)].copy()
    for frame in (held, live):
        frame["switch_age"] = frame.apply(switch_age, axis=1)
    return held, live


def tree_numbers(fit_rows: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """The tree's final (calibrated) number for `rows`, calibrated on `fit_rows` only."""
    calibrator = m.Calibrator(False).fit(fit_rows["before_signing_oos"], fit_rows, fit_rows["target_before_signing"])
    return calibrator.predict(rows["before_signing_oos"], rows)


def main() -> int:
    held, live = load()
    draws = fit(held, tree_numbers(held, held))
    X = signals(held)
    print(f"learned from {held[['player_id', 'commitment_date']].drop_duplicates().shape[0]:,} commitments (classes {CLASSES[0]}-{CLASSES[-1]}); "
          f"snapshots with a fresh switch: {int(X[:, 0].sum())}, old switch: {int(X[:, 1].sum())}")
    print(f"  fresh insider switch (under {FRESH_DAYS} days):  {multiplier(draws[:, 0])}")
    print(f"  old insider switch ({FRESH_DAYS}+ days):         {multiplier(draws[:, 1])}")
    prob = predict(live, live["before_signing_full"].to_numpy(), draws)
    live["bayes_median"], live["bayes_low"], live["bayes_high"] = np.median(prob, 1), np.percentile(prob, 10, 1), np.percentile(prob, 90, 1)
    live[["class_year", "player_id", "name", "committed_school", "commitment_date", "before_signing_full", "switch_age",
          "bayes_median", "bayes_low", "bayes_high"]].to_csv("data/cache/bayes_live.csv", index=False)
    update_board(live, draws)
    moved = live[~live["switch_age"].isna()].sort_values("bayes_median", ascending=False)
    print(f"\nlive commitments with an insider switch: {len(moved)}")
    for r in moved.head(12).itertuples():
        print(f"  {r.name:24s} {r.committed_school:28s} tree {r.before_signing_full:4.0%} -> {r.bayes_median:4.0%} "
              f"(80% range {r.bayes_low:.0%}-{r.bayes_high:.0%}), switch {r.switch_age:.0f} days old")
    return 0


BOARD = Path("data/flip_boards/live_flip_risk.csv")


def update_board(live: pd.DataFrame, draws: np.ndarray) -> None:
    """Put the adjusted number on the flip board for players with an insider switch; everyone else keeps
    the tree's. The tree's own number stays alongside as `tree_before_signing`."""
    board = pd.read_csv(BOARD)
    key = ["class_year", "player_id", "commitment_date"]
    adjusted = live[~live["switch_age"].isna()][key + ["switch_age", "bayes_median", "bayes_low", "bayes_high"]].copy()
    adjusted["player_id"] = adjusted["player_id"].astype(str)
    board["player_id"] = board["player_id"].astype(str)
    board = board.drop(columns=[c for c in ("switch_age", "bayes_median", "bayes_low", "bayes_high", "tree_before_signing", "insider_switch_adjusted") if c in board])
    board = board.merge(adjusted, on=key, how="left")
    board["tree_before_signing"] = board["break_before_signing"]
    hit = board["bayes_median"].notna()
    # Start from the board's own number (the middle of the tree ensemble) and its range; the adjusted range
    # widens by the multiplier's own uncertainty (its 10th and 90th percentiles at the low and high ends).
    age = board.loc[hit, "switch_age"].to_numpy()
    effect = np.where(age < FRESH_DAYS, 0, 1)
    shift = lambda q: np.percentile(draws, q, axis=0)[effect]
    adjust = lambda p, q: 1 / (1 + np.exp(-(logit(p) + shift(q))))
    board.loc[hit, "break_before_signing"] = adjust(board.loc[hit, "tree_before_signing"], 50).round(4)
    board.loc[hit, "before_signing_low"] = adjust(board.loc[hit, "before_signing_low"], 10).round(4)
    board.loc[hit, "before_signing_high"] = adjust(board.loc[hit, "before_signing_high"], 90).round(4)
    fresh, old = np.median(np.exp(draws[:, 0])), np.median(np.exp(draws[:, 1]))
    board["insider_switch_adjusted"] = np.where(hit, board["switch_age"].map(
        lambda age: f"insider switched away {age:.0f} days ago: odds x{fresh if age < FRESH_DAYS else old:.1f}" if pd.notna(age) else ""), "")
    board = board.drop(columns=["switch_age", "bayes_median", "bayes_low", "bayes_high"]).sort_values("break_before_signing", ascending=False)
    board.to_csv(BOARD, index=False)
    print(f"flip board: {int(hit.sum())} commitments adjusted for an insider switch (the tree's number kept as tree_before_signing)")


# ---- the test: leave one class out ----------------------------------------------------------------------
def cross_check() -> int:
    from sklearn.metrics import brier_score_loss, roc_auc_score
    held, _ = load()
    rows = []
    for check_year in CLASSES:
        fit_rows, check = held[held["class_year"] != check_year], held[held["class_year"] == check_year].copy()
        draws = fit(fit_rows, tree_numbers(fit_rows, fit_rows), draws=3000, burn=2000, chains=2)
        check["tree"] = tree_numbers(fit_rows, check)
        prob = predict(check, check["tree"].to_numpy(), draws)
        check["bayes"], check["low"], check["high"] = np.median(prob, 1), np.percentile(prob, 10, 1), np.percentile(prob, 90, 1)
        y, board = check["target_before_signing"], check.assign(target=check["target_before_signing"])
        for label, col in (("tree alone", "tree"), ("tree + Bayesian", "bayes")):
            rows.append({"test class": check_year, "model": label, "ranking (AUC)": round(roc_auc_score(y, check[col]), 4),
                         "error (Brier)": round(brier_score_loss(y, check[col]), 4), "weekly top 25 that broke": m.board_backtest(board, col)["top_25_hit_rate"]})
        age = check["switch_age"]
        print(f"test class {check_year} (learned fresh {multiplier(draws[:, 0])}, old {multiplier(draws[:, 1])}):")
        for group, mask in (("fresh insider switch", age < FRESH_DAYS), (f"old insider switch ({FRESH_DAYS}+ days)", age >= FRESH_DAYS)):
            g = check[mask].sort_values("snapshot_date").drop_duplicates(["player_id", "commitment_date"], keep="last")
            if len(g):
                inside = "inside" if g.low.mean() <= g.target_before_signing.mean() <= g.high.mean() else "OUTSIDE"
                print(f"   {group:30s} {len(g):3d} players | actually flipped {g.target_before_signing.mean():4.0%} | tree said {g.tree.mean():4.0%} | "
                      f"Bayesian said {g.bayes.mean():4.0%} (range {g.low.mean():.0%}-{g.high.mean():.0%}, actual {inside})")
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(cross_check() if "--check" in sys.argv else main())
