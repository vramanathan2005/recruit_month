#!/bin/zsh
# Daily flip refresh: new 247 timelines, the day's web report visits, insider coverage and coaching news,
# then the flip models, then the upload to Tars's Recruiting tab. Run by launchd once a day at 21:00 (4-5 hours, so clear of the
# 4:00 and 6:45 web reports, which also read 247; it uses that morning's web report), so news like a coach's firing or a decommit shows the same day (com.tars.flip-refresh in ~/Library/LaunchAgents); safe to run by hand.
#
# A data step that fails falls back to the last good copy of its data; a model step that fails stops the
# run before the upload, so Tars keeps yesterday's numbers rather than half-finished ones.
# On3 insider picks aren't refreshed here (that reads every recruit's page in a signed-in browser, hours of
# work): see backfill_predictions.py in ~/Tars/reports/web_report.

cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
# One run a day (three set off 247's bot check, Nov 5) and each takes 4-5 hours: a run that starts while the last is still going
# skips rather than racing it over the same files.
LOCK=data/.flip-refresh.pid
if [[ -f $LOCK ]] && kill -0 "$(cat $LOCK)" 2>/dev/null; then
  echo "=== flip refresh $(date): the previous run (pid $(cat $LOCK)) is still going; skipping ==="
  exit 0
fi
echo $$ > $LOCK
trap 'rm -f $LOCK' EXIT
echo "=== flip refresh $(date) ==="
# Cloudflare's browser login lasts an hour; a scheduled run finds it expired, and the database and KV commands fail
# rather than renew it. Asking who's logged in renews it.
npx --prefix "$HOME/Tars" wrangler whoami >/dev/null 2>&1 || true

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
# 2026 has signed; it can't flip. Fatal: a scrape that couldn't load the live classes (247's bot check) leaves the
# files as they were, and the models shouldn't run as if that were today's board
# Chrome keeps 247's browser state; requests is available as an explicit fallback.
if [[ ${TARS_247_TRANSPORT:-browser} == browser ]]; then
  step fatal /opt/homebrew/bin/python3 "$HOME/Tars/scripts/recruiting/browser_pull.py" --refresh-years 2027 2028
else
  step fatal $PY scrape_247_commit_dates.py --refresh-years 2027 2028
fi
step ok    $PY apply_tars_corrections.py             # staff commitment corrections from Tars, over the 247 data just scraped
step ok    $PY build_player_image_index.py           # headshots for newly ranked recruits, from the pages just scraped
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
