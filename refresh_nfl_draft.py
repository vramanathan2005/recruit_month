"""NFL draft picks for the Recruiting map's draft view (Tars: Recruiting → Map → NFL Draft), from 2020 to the latest draft:
College Football Data's /draft/picks, one request a year, with each pick's college and hometown, into
data/nfl_draft_history.csv (the columns export_map_for_tars.py reads). Re-asks only from April to June (the draft and
the weeks after it, when hometowns get filled in) or when the latest year is missing.

    .venv/bin/python refresh_nfl_draft.py [--force]
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

FIRST = 2020
OUT = Path("data/nfl_draft_history.csv")
TARS = Path.home() / "Tars"
COLUMNS = ["CollegeAthleteId", "NflAthleteId", "CollegeId", "CollegeTeam", "CollegeConference", "NflTeamId", "NflTeam", "Year", "Overall",
           "Round", "Pick", "Name", "Position", "Height", "Weight", "PreDraftRanking", "PreDraftPositionRanking", "PreDraftGrade",
           "HometownInfo City", "HometownInfo State", "HometownInfo Country", "HometownInfo Latitude", "HometownInfo Longitude", "HometownInfo CountyFips"]


def key() -> str:
    for line in (TARS / ".dev.vars").read_text().splitlines():
        if line.startswith("CFBD_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'")
    sys.exit("No CFBD_API_KEY in Tars's .dev.vars")


def picks(year: int, token: str) -> list[dict]:
    req = urllib.request.Request(f"https://api.collegefootballdata.com/draft/picks?year={year}",
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/json", "User-Agent": "tars-recruiting-map"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def row(p: dict) -> dict:
    h = p.get("hometownInfo") or {}
    return {"CollegeAthleteId": p.get("collegeAthleteId"), "NflAthleteId": p.get("nflAthleteId"), "CollegeId": p.get("collegeId"),
            "CollegeTeam": p.get("collegeTeam"), "CollegeConference": p.get("collegeConference"), "NflTeamId": p.get("nflTeamId"),
            "NflTeam": p.get("nflTeam"), "Year": p.get("year"), "Overall": p.get("overall"), "Round": p.get("round"), "Pick": p.get("pick"),
            "Name": p.get("name"), "Position": p.get("position"), "Height": p.get("height"), "Weight": p.get("weight"),
            "PreDraftRanking": p.get("preDraftRanking"), "PreDraftPositionRanking": p.get("preDraftPositionRanking"), "PreDraftGrade": p.get("preDraftGrade"),
            "HometownInfo City": h.get("city"), "HometownInfo State": h.get("state"), "HometownInfo Country": h.get("country"),
            "HometownInfo Latitude": h.get("latitude"), "HometownInfo Longitude": h.get("longitude"), "HometownInfo CountyFips": h.get("countyFips")}


if __name__ == "__main__":
    today = date.today()
    latest = today.year if today.month >= 5 else today.year - 1          # the draft is in late April
    have = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8-sig") as f:
            have = {int(r["Year"]) for r in csv.DictReader(f) if r.get("Year")}
    if "--force" not in sys.argv and latest in have and not 4 <= today.month <= 6:
        print(f"draft picks {FIRST}-{latest} on hand")
        sys.exit(0)
    token = key()
    rows = []
    for year in range(FIRST, latest + 1):
        got = picks(year, token)
        rows += [row(p) for p in got]
        print(year, len(got), "picks,", sum(1 for p in got if (p.get("hometownInfo") or {}).get("latitude")), "with a hometown")
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows):,} picks → {OUT}")
