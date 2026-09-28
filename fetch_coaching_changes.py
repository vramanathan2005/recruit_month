#!/usr/bin/env python3
"""FBS head coaching changes (firings, resignations, departures) with dates, from Wikipedia's season pages.

The flip snapshot model uses these for "his school's head coach left since he committed". Hot-seat
rankings would be the earlier warning, but they have no usable history to test against; actual changes do.

    python3 fetch_coaching_changes.py [--years 2021 2022 ...]   → data/coaching_changes.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path

from score_commitment_strength import norm_team

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "TarsRecruitingResearch/1.0 (coaching changes for a recruiting model)"}
# Wikipedia's school names → the form norm_team() gives 247's names.
ALIASES = {"miami (fl)": "miami", "miami (oh)": "miami (oh)", "ole miss": "ole miss", "usf": "usf", "south florida": "usf",
           "ucf": "ucf", "smu": "smu", "tcu": "tcu", "byu": "byu", "lsu": "lsu", "uab": "uab", "utsa": "utsa", "utep": "utep",
           "unlv": "unlv", "fiu": "fiu", "fau": "florida atlantic", "florida atlantic": "florida atlantic", "nc state": "nc state",
           "hawaii": "hawaii", "hawaiʻi": "hawaii", "san jose state": "san jose state", "san josé state": "san jose state",
           "louisiana–monroe": "ul monroe", "louisiana-monroe": "ul monroe", "ulm": "ul monroe", "umass": "massachusetts",
           "uconn": "connecticut", "southern miss": "southern miss", "texas a&m": "texas a&m", "app state": "appalachian state"}


def clean(text: str) -> str:
    return re.sub(r"\s*\[\s*\w+\s*\]", "", re.sub(r"\s+", " ", text)).strip()


def school_key(name: str) -> str:
    base = clean(name).lower()
    return ALIASES.get(base, norm_team(base))


def timeline_school_keys(commit_dates: Path, timeline: list[dict]) -> dict[str, str]:
    """The timeline names schools with mascots ("Miami (OH) RedHawks"); pair each recruit's short committed
    school from commit_dates.csv with his timeline commitment to learn norm_team(timeline name) → school_key."""
    from collections import Counter, defaultdict
    short = {(r["class_year"], r["player_id"]): r["committed_team"] for r in csv.DictReader(commit_dates.open(encoding="utf-8")) if r.get("committed_team")}
    pairs: dict[str, Counter] = defaultdict(Counter)
    for event in timeline:
        name = short.get((event["class_year"], event["player_id"]))
        if event["event_type"] != "Commitment" or not name:
            continue
        long_name, key = norm_team(event["school"]), school_key(name)
        if long_name == key or long_name.startswith(key + " "):
            pairs[long_name][key] += 1
    return {long_name: counts.most_common(1)[0][0] for long_name, counts in pairs.items()}


def parse_date(text: str) -> str:
    text = clean(text)
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    return ""


def changes(year: int) -> list[dict]:
    import requests  # only the download needs these; the flip model imports school_key() from here
    from bs4 import BeautifulSoup

    response = requests.get(API, params={"action": "parse", "page": f"{year} NCAA Division I FBS football season", "prop": "text",
                                         "format": "json", "formatversion": 2}, headers=HEADERS, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.json()["parse"]["text"], "html.parser")
    rows = []
    for table in soup.select("table"):
        first = table.select_one("tr")
        head = [clean(c.get_text(" ")).lower() for c in first.select("th,td")] if first else []
        if not head or "outgoing coach" not in head or "date" not in head:
            continue
        kind = "off-season" if "previous position" in head else "in-season"
        col = {name: head.index(name) for name in head}
        for tr in table.select("tr")[1:]:
            cells = [clean(c.get_text(" ")) for c in tr.select("td,th")]
            if len(cells) < len(head):
                continue
            school = cells[col.get("school", col.get("team", 0))]
            rows.append({
                "season": year, "kind": kind, "school": school, "school_key": school_key(school),
                "outgoing_coach": cells[col["outgoing coach"]], "date": parse_date(cells[col["date"]]),
                "reason": cells[col["reason"]] if "reason" in col else "", "replacement": cells[col["replacement"]] if "replacement" in col else "",
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2020, datetime.now().year + 1)))
    parser.add_argument("--out", type=Path, default=Path("data/coaching_changes.csv"))
    args = parser.parse_args()
    rows = []
    for year in args.years:
        found = changes(year)
        print(f"{year}: {len(found)} coaching changes ({sum(1 for r in found if not r['date'])} without a date)")
        rows += found
        time.sleep(1)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
