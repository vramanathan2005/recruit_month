#!/bin/zsh
# Daily flip refresh: new 247 timelines, the day's web report visits, insider coverage and coaching news,
# then the flip models, then the upload to Tars's Recruiting tab. Run by launchd every morning at 9:30,
# after the 8:30 web report (com.tars.flip-refresh in ~/Library/LaunchAgents); safe to run by hand.
#
# A data step that fails falls back to the last good copy of its data; a model step that fails stops the
# run before the upload, so Tars keeps yesterday's numbers rather than half-finished ones.
# On3 insider picks aren't refreshed here (that reads every recruit's page in a signed-in browser, hours of
# work): see backfill_predictions.py in ~/Tars/reports/web_report.

cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
echo "=== flip refresh $(date) ==="

step() {   # step <fatal|ok> <command...>: run it, log how long it took, stop on a fatal failure
  local fatal=$1; shift
  local start=$SECONDS
  echo "-- $*"
  "$@"
  local rc=$?
  echo "   exit $rc after $(( SECONDS - start ))s"
  if (( rc != 0 )) && [[ $fatal == fatal ]]; then
    echo "!! stopping before the upload; Tars keeps the previous numbers"
    exit $rc
  fi
}

# Data (each keeps its previous output if it fails)
step ok    $PY scrape_247_commit_dates.py --refresh-years 2027 2028   # 2026 has signed; it can't flip
step ok    $PY score_commitment_strength.py
step ok    $PY extract_web_report_events.py          # OpenAI, only the new articles (about a cent a day)
step ok    $PY extract_insider_signals.py
step ok    $PY fetch_coaching_changes.py
step ok    $PY fetch_hot_seat_history.py

# Models, then the upload
step fatal $PY train_flip_snapshot_model.py
step fatal $PY bayes_layer.py
step fatal $PY train_flip_destination_model.py
step ok    $PY train_commit_destination_model.py   # where uncommitted recruits commit (Commit Watch); keeps yesterday's if it fails
step fatal $PY export_for_tars.py

echo "=== done in $(( SECONDS / 60 ))m $(( SECONDS % 60 ))s ==="
