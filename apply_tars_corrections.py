"""Staff commitment corrections from Tars, applied to the 247 data before the flip models run.

Tars staff correct a commitment when 247 has it wrong (Tars: lib/db.ts player_commitments, "Correct commitment" on a
player's Recruiting card): a recruit who "reopened his recruitment" but is still committed, which 247 logs as a
decommit, or a decommit 247 hasn't caught. Each morning's scrape rewrites data/timeline_events.csv from 247, so this
runs right after it (daily_flip_refresh.sh) and lays the corrections back over:

  committed to X   the commitment and decommitment events after his last commitment to X are dropped, so he is a
                   live X commit again (one is added on the correction's date if 247 never had him committed to X),
                   and data/commit_dates.csv names X as his school
  uncommitted      a decommitment from his school is added on the correction's date, if 247 still has him committed

A correction is matched to a 247 recruit by name and class (and high school when two share a name); one that matches
no one, or more than one, is reported and skipped. With no corrections, or Tars unreachable, nothing changes.

    .venv/bin/python apply_tars_corrections.py            # apply
    .venv/bin/python apply_tars_corrections.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from score_commitment_strength import norm_team

TARS = Path.home() / "Tars"
TIMELINE = Path("data/timeline_events.csv")
COMMITS = Path("data/commit_dates.csv")
RANKINGS = Path("data/raw_rankings.csv")
COMMIT_TYPES = {"Commitment", "Decommit"}


def bare(name: str) -> str:
    name = re.sub(r"\b(?:jr|sr|ii|iii|iv|v)\b\.?", "", (name or "").lower())
    return re.sub(r"[^a-z]", "", name)


def parse_day(text: str):
    try:
        return datetime.strptime(text.strip(), "%m/%d/%Y").date()
    except (ValueError, AttributeError):
        return None


def mdy(day) -> str:
    return f"{day.month}/{day.day}/{day.year}"


def corrections() -> list[dict]:
    """The standing corrections, from Tars's database (an empty list if it can't be read)."""
    sql = ("SELECT player_key, school, note, source, name, class_year, high_school, added_by_name, added_at "
           "FROM player_commitments WHERE removed_at IS NULL")
    try:
        out = subprocess.run(["npx", "wrangler", "d1", "execute", "tars-d1", "--remote", "--json", "--command", sql],
                             cwd=TARS, capture_output=True, text=True, timeout=120)
        return json.loads(out.stdout)[0].get("results", []) if out.returncode == 0 else []
    except (subprocess.SubprocessError, json.JSONDecodeError, IndexError, OSError):
        return []


def read(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    fixes = [c for c in corrections() if c.get("name") and c.get("class_year")]
    if not fixes:
        print("no Tars commitment corrections")
        return 0

    _, rankings = read(RANKINGS)
    tl_fields, timeline = read(TIMELINE)
    cd_fields, commits = read(COMMITS)
    changed_tl = changed_cd = False

    for fix in fixes:
        cls, who = str(fix["class_year"]), bare(fix["name"])
        found = {r["player_id"]: r for r in rankings if r["class_year"] == cls and bare(r["name"]) == who}
        if len(found) > 1 and fix.get("high_school"):
            school_word = (re.findall(r"[a-z]+", fix["high_school"].lower()) or [""])[0]
            found = {pid: r for pid, r in found.items() if school_word and school_word in r.get("high_school", "").lower()}
        if len(found) != 1:
            print(f"  skipped {fix['name']} ({cls}): {'no' if not found else len(found)} 247 match{'es' if len(found) > 1 else ''}")
            continue
        pid, ranked = next(iter(found.items()))
        on = datetime.fromtimestamp(int(fix["added_at"]) / 1000, tz=timezone.utc).date()
        events = sorted((e for e in timeline if e["player_id"] == pid and e["event_type"] in COMMIT_TYPES),
                        key=lambda e: parse_day(e["event_date"]) or on)
        wanted = norm_team(fix["school"] or "")
        if wanted:
            to_school = [e for e in events if e["event_type"] == "Commitment" and norm_team(e["school"]) == wanted]
            if to_school:
                since = parse_day(to_school[-1]["event_date"])
                drop = [e for e in events if (parse_day(e["event_date"]) or on) > since]
                school_name, committed_on = to_school[-1]["school"], to_school[-1]["event_date"]
            else:
                drop, school_name, committed_on = [e for e in events if (parse_day(e["event_date"]) or on) > on], fix["school"], mdy(on)
                timeline.append({"class_year": cls, "player_id": pid, "name": ranked["name"], "event_type": "Commitment",
                                 "event_date": committed_on, "school": school_name})
                changed_tl = True
            if drop:
                ids = {id(e) for e in drop}
                timeline[:] = [e for e in timeline if id(e) not in ids]
                changed_tl = True
            row = next((r for r in commits if r["player_id"] == pid and r["class_year"] == cls), None)
            if row is None:
                row = {f: ranked.get(f, "") for f in cd_fields}
                commits.append(row)
            if norm_team(row.get("committed_team", "")) != wanted or not row.get("committed_date"):
                row.update({"committed_team": school_name, "commitment_type": row.get("commitment_type") or "Commit",
                            "committed_date": committed_on, "commit_date_source": "tars_correction"})
                changed_cd = True
            print(f"  {ranked['name']} ({cls}): committed to {school_name} since {committed_on}"
                  f"{f'; dropped {len(drop)} later 247 event(s)' if drop else ''} · {fix.get('note') or fix['source']} ({fix.get('added_by_name') or 'staff'})")
        else:
            if events and events[-1]["event_type"] == "Commitment":
                timeline.append({"class_year": cls, "player_id": pid, "name": ranked["name"], "event_type": "Decommit",
                                 "event_date": mdy(on), "school": events[-1]["school"]})
                changed_tl = True
                print(f"  {ranked['name']} ({cls}): decommitted from {events[-1]['school']} {mdy(on)} · {fix.get('note') or fix['source']}")
            else:
                print(f"  {ranked['name']} ({cls}): already uncommitted on 247")

    if args.dry_run:
        print("dry run: nothing written")
        return 0
    if changed_tl:
        write(TIMELINE, tl_fields, timeline)
    if changed_cd:
        write(COMMITS, cd_fields, commits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
