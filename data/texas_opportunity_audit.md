# Texas Opportunity Audit

Audit date: 2026-06-02

## What Was Checked

Reviewed the top 25 rows in `data/flip_boards/target_board_texas_2027.csv` after the Texas Opportunity board was separated from the national flip-risk board.

The audit checked:

- whether `Attack Now` had a real Texas reason
- whether the player had post-commit activity
- whether the Texas angle was in-state, nearby, or tied to a regional/rival committed school
- whether broad regional geography alone was creating noisy `Attack Now` labels

## Issue Found

The old rule allowed broad `regional` distance plus activity to create `Attack Now`.

That made a few rows look too aggressive even though the only Texas angle was loose geography, not a true Texas-specific signal. Examples that were too noisy:

- Deng Tong, Kansas, home MO
- David Rushing, Kansas State, home AZ
- Donivan Moore, Auburn, home AL
- Roye Oliver III, USC, home AZ
- Jai Jones, Wisconsin, home AZ

Those players may still be worth monitoring, but they should not sit at the top of a Texas staff action board without a stronger Texas reason.

## Fix Made

`Attack Now` now requires:

- post-commit activity, and
- either in-state/nearby Texas geography or a regional/rival committed school

Broad `regional` proximity alone now becomes `Monitor`, not `Attack Now`.

## Result After Fix

Texas Opportunity buckets:

- `Monitor`: 152
- `Attack Now`: 73
- `Strong Texas Angle`: 25

Former noisy regional-only examples were demoted to `Monitor`.

The top 25 now mostly consists of:

- Texas in-state recruits with activity
- commits to Texas regional/rival schools
- blue-chip regional/rival players with activity

## Remaining Caveat

This is still based on public/captured 247 data. If the staff knows Texas is actively recruiting a player, that should be added through a staff override file in the next pass.
