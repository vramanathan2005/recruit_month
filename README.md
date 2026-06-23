# 247 Commitment Date Scrape

This workspace builds a 247Sports-only dataset for football recruits in the
2022-2027 classes.

## Data Definition

- Player pool: `CompositeRecruitRankings` with `InstitutionGroup=HighSchool`
- Commitment dates: each player's 247Sports recruitment detail page
  `commitmentBlock`
- Fallback commitment dates: 247Sports class-level `Football Commits` pages
- Join key: 247 player ID from the profile URL
- Excluded as a date source: Transfer Portal pages and current NCAA player
  profile state

That means a player like Dante Moore is treated as a 2023 high-school recruit,
  and the scrape looks for the high-school recruitment detail `commitmentBlock`
  rather than his later Oregon transfer/current-team state.

## Run

```bash
python3 scrape_247_commit_dates.py
```

Faster primary-source run:

```bash
python3 scrape_247_commit_dates.py --skip-commit-pages --detail-workers 8
```

Useful debug run:

```bash
python3 scrape_247_commit_dates.py --years 2023 --max-pages 2
```

## Dashboard

Install app dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
.venv/bin/streamlit run streamlit_app.py
```

## Outputs

- `data/commit_dates.csv`: joined recruit-level dataset
- `data/raw_rankings.csv`: raw high-school ranking rows
- `data/raw_commits.csv`: raw class commit rows
- `data/timeline_events.csv`: 247 timeline events, including commits,
  decommits, signings, enrollments, offers, and official/unofficial visits
- `data/summary_by_day.csv`: commitment counts by date
- `data/summary_by_week.csv`: commitment counts by week
- `data/summary_by_month.csv`: commitment counts by month
- `data/summary.md`: quick June/July/July 4/August 1 summary
- `data/commitment_strength_scores.csv`: rules-based commitment strength score
- `data/commitment_strength_summary.md`: score distribution and weakest commitments
- `data/commitment_strength_backtest_events.csv`: commitment-event backtest rows
- `data/commitment_strength_backtest_buckets.csv`: score bucket outcome rates
- `data/commitment_strength_backtest_by_position.csv`: outcome rates by position
- `data/commitment_strength_backtest_summary.md`: readable backtest summary
- `data/commitment_strength_holdout_samples.csv`: train/holdout/live-ish validation
- `data/commitment_strength_holdout_buckets.csv`: bucket rates by validation split
- `data/commitment_strength_holdout_summary.md`: readable holdout summary
- `data/flip_model_validation.csv`: trained model train/holdout validation
- `data/flip_model_probability_buckets.csv`: modeled probability calibration buckets
- `data/flip_model_events.csv`: event-level model predictions for known outcomes
- `data/flip_boards/`: staff-facing weakest-commitment boards
- `data/flip_boards/boss_view_*.csv`: simplified staff/boss exports with
  modeled flip/stick probabilities
- `data/flip_boards/target_board_texas_2027.csv`: Texas-specific 2027 flip
  target board, excluding existing Texas commits and signed/enrolled players
- `data/texas_staff_overrides.csv`: optional staff-maintained Texas signal
  file used to promote/hide players and add notes to the Texas board

Cached HTML is stored under `data/cache/` so repeat runs are faster.

## Commitment Strength V1

Run:

```bash
python3 score_commitment_strength.py
```

The V1 score is a transparent rules-based vulnerability index. The base score
uses pre-outcome signals so it can be backtested honestly:

- commitment timing relative to early signing day
- prior commitments and decommitments from the timeline
- pre-commit offer volume and visit pattern
- same-school, same-position class crowding
- higher-rated commits at the same school/position
- high-school city to campus distance where available, with state fallback
- rating tier as an attention/volatility proxy

The output is `commitment_strength_score`, where `100` is strongest and `0` is
weakest. The current weights were adjusted against the event-level historical
backtest.

The output also includes `live_commitment_strength_score`, which adjusts the
base score for post-commit recruiting activity captured by 247:

- other-school offers after commitment
- committed-school visits after commitment
- other-school official/unofficial visits after commitment
- post-commit decommit/commitment activity involving other schools

Backtest:

```bash
python3 backtest_commitment_strength.py
```

The current backtest buckets after adding 2027 and timeline offer/visit
extraction:

- `Weak / Attack`: 52.1% decommit/flip rate
- `Vulnerable`: 31.0% decommit/flip rate
- `Stable / Monitor`: 16.2% decommit/flip rate
- `Strong`: 6.5% decommit/flip rate

Coach stability, hot-seat status, and social activity still need separate
enrichment tables before calling the score a full model.

## Trained Flip Model

Run after `score_commitment_strength.py` and before building boards:

```bash
.venv/bin/python train_flip_model.py
```

This trains a calibrated logistic regression model on known commitment-event
outcomes. It uses classes `2022-2024` for training, checks `2025` as the main
holdout, and reports `2026` as a live/current-ish sanity check.

The model writes two primary probability columns into
`data/commitment_strength_scores.csv`:

- `model_decommit_flip_probability`
- `model_stick_probability`

Current validation:

- train classes `2022-2024`: AUC `0.7541`
- holdout class `2025`: AUC `0.7107`
- live/current-ish class `2026`: AUC `0.7163`

The old bucket probabilities remain in the files as historical context, but
the boards and dashboard use the trained model probability as the primary
flip/decommit estimate.

Holdout validation:

```bash
python3 validate_commitment_strength.py
```

The current holdout results:

- train classes `2022-2024`: AUC `0.7171`
- holdout class `2025`: AUC `0.6978`
- live/current-ish class `2026`: AUC `0.6838`

Flip board exports:

```bash
python3 build_flip_boards.py
```

This writes sortable CSVs under `data/flip_boards/`, including all commits,
current-class commits, power-team guess commits, and top 25 weakest commits by
position.
The board sorts by `model_decommit_flip_probability` and includes
`commitment_type`, `is_signed_or_enrolled`, geography, offer, visit, and
rules-based live score columns so staff can filter out players who are no
longer live flip targets and quickly see distance plus current recruiting
activity risk.

Geography uses the city/state embedded in 247's high-school label when possible
and falls back to state centroids only when city lookup fails. City coordinates
come from `geonamescache` plus the local Census place gazetteer at
`data/geo/2024_Gaz_place_national.zip`. Major programs use campus-city
coordinates; other programs fall back to their state centroid. The output
includes `home_city`, `home_state`, `home_to_school_miles`, `distance_bucket`,
and `geography_precision` so staff can see whether a distance is city-level or
fallback.

The simplified boss-view files are:

- `data/flip_boards/boss_view_all_commits.csv`
- `data/flip_boards/boss_view_2027_commits.csv`
- `data/flip_boards/boss_view_power_team_commits.csv`
- `data/flip_boards/boss_view_2027_power_team_commits.csv`

These keep the action columns only: player, position, committed school,
live/base score, modeled decommit/flip probability, modeled stick
probability, distance, post-commit offers/visits, reason flagged, and profile
URL.

The Texas target board is:

- `data/flip_boards/target_board_texas_2027.csv`

It filters to 2027 power-team commits, excludes recruits already committed to
Texas, excludes signed/enrolled players, and adds `home_to_target_miles` plus
`target_distance_bucket` using high-school city to Austin distance when
available. It sorts by the
Texas opportunity label first, then trained model flip/decommit probability.
It adds `texas_context_tags` for Texas-relevant context such as in-state
recruits, regional/rival commits, 4/5-star recruits, and players with
post-commit other-school activity.

Optional staff overrides live in:

- `data/texas_staff_overrides.csv`

The file is created automatically by `build_flip_boards.py` if it does not
exist. Fill it by `profile_url` with these columns:

- `staff_priority`
- `texas_offer`
- `texas_visit`
- `texas_recruiting`
- `do_not_pursue`
- `staff_notes`

Use `yes`/`no` style values. `staff_priority=yes` promotes a player to the top
of the Texas opportunity board and marks him `Attack Now`. `texas_offer`,
`texas_visit`, or `texas_recruiting` add Texas signal and can move a public-data
`Monitor`/`Need Texas Signal` player into `Strong Texas Angle`. `do_not_pursue`
hides the player from the Texas board. `staff_notes` show in the dashboard hover
profile.
