"""Export the Recruiting map's data for Tars (Recruiting → Map, Nov 16 2026).

Every recruit in a class as a dot at his hometown, with what any school did with him, so the map can filter by any
team, any conference or any set of them:

    recruit-map:v1:schools          every school a recruit heard from: name, logo, campus, conference by season
    recruit-map:v1:<class>          247's recruits (Composite-ranked, and committed ones outside it): hometown, stars,
                                    rank, who offered him, who had him on an official or unofficial visit, where he
                                    committed and where he signed or enrolled
    recruit-map:v1:draft            every NFL draft pick from 2015 (data/nfl_draft_history.csv, College Football Data):
                                    his hometown, college, year, round and pick
    recruit-map:v1:<class>:uc       the high school players Tars knows who aren't one of those 247 recruits (Tars's
    recruit-map:v1:<class>:mp       recruits:v1 tables), at their school's town: UC Report's prospects (on the map by
                                    default), and the MaxPreps-only roster players (a checkbox)

Hometowns are placed from 247's "School (City, ST)" and the high schools' city and state (geography_features.py: city
coordinates, else the state's center, marked approximate). Conferences come from Tars's
research/recruiting_map/conferences.json (ESPN's standings by season, scripts/recruiting/map-conferences.mts).

    .venv/bin/python export_map_for_tars.py [--dry-run]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from export_for_tars import NAMESPACE_ID, TARS, school
from geography_features import STATE_CENTROIDS, coords_for_city_state, coords_for_team, recruit_location_from_high_school

CLASSES = list(range(2022, datetime.now().year + 5))           # 2022 (the timelines' first class) to the youngest in high school
HS_CLASSES_FROM = datetime.now().year + 1                      # classes still in high school: Tars has their rosters
OUT = Path("data/cache/tars-map")
FINAL = ("Enrollment", "Signing")


def place(city: str, state: str) -> tuple[float, float, int] | None:
    """(longitude, latitude, exact: 1 city / 0 the state's center)."""
    state = (state or "").strip().upper()
    hit = coords_for_city_state(city, state) if city and state else None
    if hit:
        return round(hit[1], 3), round(hit[0], 3), 1
    centre = STATE_CENTROIDS.get(state)
    return (centre[1], centre[0], 0) if centre else None


def num(value):
    try:
        v = float(value)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def stars_of(rating: float | None, score247: float | None = None) -> int | None:
    if rating:
        return 5 if rating >= 0.9834 else 4 if rating >= 0.89 else 3 if rating >= 0.797 else 2
    if score247:
        return 5 if score247 >= 98 else 4 if score247 >= 90 else 3 if score247 >= 80 else 2
    return None


def kv_get(key: str) -> bytes | None:
    p = subprocess.run(["npx", "wrangler", "kv", "key", "get", key, "--namespace-id", NAMESPACE_ID, "--remote"], cwd=TARS, capture_output=True)
    return p.stdout if p.returncode == 0 and p.stdout else None


def build() -> list[dict]:
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rankings = pd.read_csv("data/raw_rankings.csv", dtype={"player_id": str})
    timeline = pd.read_csv("data/timeline_events.csv", dtype={"player_id": str})
    commits = pd.read_csv("data/raw_commits.csv", dtype={"player_id": str}) if Path("data/raw_commits.csv").exists() else pd.DataFrame()
    conferences = json.loads((TARS / "research/recruiting_map/conferences.json").read_text())
    # 247's headshots (build_player_image_index.py), for the map's list and cards
    images = dict(pd.read_csv("data/player_images.csv", dtype={"player_id": str}).values) if Path("data/player_images.csv").exists() else {}
    colors = json.loads((TARS / "research/recruiting_map/colors.json").read_text()) if (TARS / "research/recruiting_map/colors.json").exists() else {}

    def color_of(espn: str | None, key: str) -> str | None:
        """The school's color for its dots (Texas's own burnt orange; a white primary gives way to its second color)."""
        if key == "texas":
            return "#bf5700"
        c = colors.get(espn or "") or {}
        main, alt = (c.get("color") or "").lower(), (c.get("alt") or "").lower()
        light = lambda h: len(h) == 7 and sum(int(h[i:i + 2], 16) for i in (1, 3, 5)) > 690
        return (alt if main and light(main) and alt and not light(alt) else main) or None

    # every school anyone heard from, once; a player's schools are indexes into it
    schools: dict[str, dict] = {}

    def school_index(raw) -> int | None:
        s = school(raw) if isinstance(raw, str) else None
        if not s:
            return None
        if s["key"] not in schools:
            espn = re.search(r"/(\d+)\.png", s.get("logo") or "")
            campus = coords_for_team(s["key"])
            schools[s["key"]] = {"key": s["key"], "name": s["name"], "logo": s.get("logo"), "espn": espn.group(1) if espn else None,
                                 "lon": round(campus[1], 3) if campus else None, "lat": round(campus[0], 3) if campus else None, "i": len(schools),
                                 "color": color_of(espn.group(1) if espn else None, s["key"])}
        return schools[s["key"]]["i"]

    events = defaultdict(list)
    for r in timeline.itertuples():
        events[r.player_id].append((str(r.event_date), r.event_type, r.school))

    def us_date(s: str) -> str:
        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", str(s or ""))
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}" if m else ""

    entries, missing = [], defaultdict(int)
    for year in CLASSES:
        rows = []
        ranked = rankings[rankings["class_year"] == year].drop_duplicates("player_id")
        extra = pd.DataFrame()
        if not commits.empty:
            extra = commits[(commits["class_year"] == year) & commits["committed_team"].fillna("").astype(str).str.strip().ne("")
                            & ~commits["player_id"].isin(ranked["player_id"])].drop_duplicates("player_id")
        for source, frame in (("ranked", ranked), ("commit", extra)):
            for r in frame.itertuples():
                pid = str(r.player_id)
                city, state = recruit_location_from_high_school(str(getattr(r, "high_school", "") or ""))
                where = place(city, state)
                if not where:
                    missing[year] += 1
                    continue
                offers, official, unofficial = set(), set(), set()
                committed = signed = None
                for day, kind, raw in sorted(events.get(pid, []), key=lambda e: us_date(e[0])):
                    i = school_index(raw)
                    if i is None:
                        continue
                    if kind == "Offer":
                        offers.add(i)
                    elif kind == "Official Visit":
                        official.add(i)
                    elif kind == "Unofficial Visit":
                        unofficial.add(i)
                    elif kind == "Commitment":
                        committed = i
                    elif kind == "Decommit" and committed == i:
                        committed = None
                    elif kind in FINAL:
                        signed = i
                if source == "commit":
                    committed = school_index(r.committed_team)
                rating = num(getattr(r, "rating", None)) if source == "ranked" else None
                hs = re.sub(r"\s*\([^()]*\)\s*$", "", str(getattr(r, "high_school", "") or "")).strip()
                rows.append([pid, str(r.name), str(r.position or ""), stars_of(rating, num(getattr(r, "rating", None)) if source == "commit" else None), rating,
                             int(num(r.national_rank)) if num(getattr(r, "national_rank", None)) else None,
                             where[0], where[1], where[2], city, state, hs,
                             signed if signed is not None else committed, committed, sorted(offers), sorted(official), sorted(unofficial),
                             images.get(pid)])
        if rows:
            entries.append({"key": f"recruit-map:v1:{year}", "value": json.dumps({
                "classYear": year, "builtAt": built_at,
                "fields": ["id", "name", "position", "stars", "rating", "rank", "lon", "lat", "exact", "city", "state", "highSchool",
                           "landed", "committed", "offers", "officialVisits", "unofficialVisits", "headshot"],
                "players": rows}, separators=(",", ":"))})
        print(f"{year}: {len(rows):,} 247 recruits on the map ({missing[year]} without a place)")

        # the high school players Tars knows (UC Report, MaxPreps) who aren't one of those 247 recruits
        if year >= HS_CLASSES_FROM:
            raw = kv_get(f"recruits:v1:{year}")
            if raw:
                data = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
                at = {f: i for i, f in enumerate(data["fields"])}
                # each high school once ([name, city, state, longitude, latitude, exact]); a player points to his
                places, school_at, hs_schools, hs_rows = {}, {}, [], []
                for row in data["rows"]:
                    if row[at["stars"]]:
                        continue                                           # linked to a 247 recruit: his 247 dot
                    i = row[at["school"]]
                    if i not in school_at:
                        name, city, state = data["schools"][i][:3]
                        if (city, state) not in places:
                            places[(city, state)] = place(city, state)
                        where = places[(city, state)]
                        school_at[i] = len(hs_schools) if where else None
                        if where:
                            hs_schools.append([name, city, state, where[0], where[1], where[2]])
                    if school_at[i] is None:
                        continue
                    hs_rows.append([row[at["ucId"]] or row[at["tarsId"]] or None, f"{row[at['first']] or ''} {row[at['last']] or ''}".strip(),
                                    row[at["position"]] or "", school_at[i], 1 if row[at["ucId"]] else 0, 1 if row[at["maxpreps"]] else 0])
                # UC Report's prospects (on the map by default) apart from the MaxPreps-only roster players (~10x as many,
                # one checkbox): each file with only the schools its players are at
                for layer, keep in (("uc", lambda row: row[4]), ("mp", lambda row: not row[4])):
                    rows_ = [row for row in hs_rows if keep(row)]
                    used = sorted({row[3] for row in rows_})
                    renum = {old: new for new, old in enumerate(used)}
                    entries.append({"key": f"recruit-map:v1:{year}:{layer}", "value": json.dumps({
                        "classYear": year, "builtAt": built_at, "schools": [hs_schools[i] for i in used],
                        "fields": ["id", "name", "position", "school", "uc", "maxpreps"],
                        "players": [row[:3] + [renum[row[3]]] + row[4:] for row in rows_]}, separators=(",", ":"))})
                print(f"{year}: {len(hs_rows):,} more high school players (UC Report / MaxPreps)")

    # every NFL draft pick from 2015, at his hometown, with his college (College Football Data's draft history)
    draft_path = Path("data/nfl_draft_history.csv")
    if draft_path.exists():
        picks = []
        for d in pd.read_csv(draft_path, dtype=str, encoding="utf-8-sig").fillna("").to_dict("records"):
            lon, lat = num(d.get("HometownInfo Longitude")), num(d.get("HometownInfo Latitude"))
            if lon is None or lat is None:
                city, state = d.get("HometownInfo City") or "", d.get("HometownInfo State") or ""
                where = place(city, state)
                if not where:
                    continue
                lon, lat = where[0], where[1]
            picks.append([d.get("Name"), d.get("Position"), school_index(d.get("CollegeTeam")), int(num(d.get("Year")) or 0), int(num(d.get("Round")) or 0),
                          int(num(d.get("Overall")) or 0), d.get("NflTeam"), round(lon, 3), round(lat, 3), d.get("HometownInfo City") or "", d.get("HometownInfo State") or ""])
        entries.append({"key": "recruit-map:v1:draft", "value": json.dumps({"builtAt": built_at, "fields": ["name", "position", "college", "year", "round", "pick",
                        "nflTeam", "lon", "lat", "city", "state"], "players": picks}, separators=(",", ":"))})
        print(f"{len(picks):,} NFL draft picks on the map")

    for s in schools.values():
        s["conference"] = {season: by.get(s["espn"]) for season, by in conferences.items() if s["espn"] and by.get(s["espn"])}
    entries.append({"key": "recruit-map:v1:schools", "value": json.dumps({"builtAt": built_at, "classes": CLASSES,
                    "schools": sorted(schools.values(), key=lambda s: s["i"])}, separators=(",", ":"))})
    print(f"{len(schools):,} schools; {sum(len(e['value']) for e in entries) / 1e6:.1f} MB in {len(entries)} KV entries")
    return entries


def upload(entries: list[dict], dry_run: bool) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failed = 0
    for entry in entries:
        path = (OUT / (entry["key"].replace(":", "-") + ".json")).resolve()
        path.write_text(entry["value"])
        if dry_run:
            continue
        p = subprocess.run(["npx", "wrangler", "kv", "key", "put", entry["key"], "--path", str(path), "--namespace-id", NAMESPACE_ID, "--remote"],
                           cwd=TARS, capture_output=True, text=True)
        failed += p.returncode != 0
    print("written to", OUT if dry_run else "KV", f"({failed} failed)" if failed else "")
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(upload(build(), args.dry_run))
