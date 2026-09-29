#!/usr/bin/env python3
"""Export the flip model's live results and every ranked recruit's recruiting history for Tars's Recruiting
tab, as Workers KV entries in Tars's `tars-prebuilt` namespace (read by lib/recruiting.ts in Tars).

    .venv/bin/python export_for_tars.py              # build and upload to the live site's KV
    .venv/bin/python export_for_tars.py --local      # load into Tars's local dev KV instead (npm run dev)
    .venv/bin/python export_for_tars.py --dry-run    # build only: data/cache/tars/recruiting-kv-*.json

Run after the models (README: "Refreshing everything"). The entries hold subscription-derived coverage and
insider picks, so they only go to KV (behind the Tars login), never to this repo: data/cache is gitignored.

Keys:
    recruiting:index            every ranked recruit in the live classes (search, and the flip badge on a profile)
    recruiting:board            every live commitment with its flip outlook (Commits and Flip Watch)
    recruiting:player:<id>      one recruit's full profile: flip outlook, timeline, insider picks, coverage
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

import train_flip_snapshot_model as m
from score_commitment_strength import norm_team

LIVE_CLASSES = [2027, 2028]
TARS = Path.home() / "Tars"
BACKFILL = TARS / "reports" / "web_report" / "backfill"
NAMESPACE_ID = "571ad7ddc4ae479198e098c611f54251"   # tars-prebuilt (Tars: scripts/prebuild/upload.ts)
OUT = Path("data/cache/tars")
MAX_BATCH_BYTES = 40 * 1024 * 1024
COVERAGE_LIMIT = 40


# ---- schools --------------------------------------------------------------------------------------------
# Our key → ESPN's key, where ESPN spells the school differently.
ESPN_KEYS = {"appalachian state": "app state", "san jose state": "san josé state", "hawaii": "hawai'i", "louisiana-monroe": "ul monroe"}
ACRONYMS = {"lsu", "tcu", "ucf", "smu", "usc", "ucla", "byu", "unlv", "utsa", "utep", "uab", "fiu", "fau", "usf", "uconn", "umass", "nc state", "liu", "utrgv"}


def espn_teams() -> dict[str, dict]:
    teams = {}
    for t in json.loads(Path("data/espn_teams.json").read_text()):
        entry = {"name": t["location"], "logo": t["logo"] or None}
        teams.setdefault(norm_team(t["display"]), entry)
        teams.setdefault(norm_team(t["location"]), entry)
    return teams


TEAMS = espn_teams()


def school(raw: str | None) -> dict | None:
    """Any school name (247 "LSU Tigers", a model key "lsu", "Florida Gators for spring game") → {name, logo}."""
    key = norm_team(raw or "")
    if not key:
        return None
    if ESPN_KEYS.get(key, key) in TEAMS:
        return {**TEAMS[ESPN_KEYS.get(key, key)], "key": key}
    pretty = key.upper() if key in ACRONYMS else re.sub(r"(^|[\s-])(\w)", lambda g: g.group(1) + g.group(2).upper(),
                                                         " ".join(w.upper() if w in ACRONYMS else w for w in key.split()))
    return {"name": pretty, "logo": None, "key": key}


# ---- plain-English reasons ------------------------------------------------------------------------------
def plural(n: str, word: str) -> str:
    return f"{n} {word}{'' if n == '1' else 's'}"


def risk_reasons(why: str, committed: str) -> list[dict]:
    """The board's `why` text → sentences, each marked as raising or lowering the risk."""
    rules = [
        (r"(\d+) official visit\(s\) elsewhere since committing", lambda g: f"Took {plural(g[0], 'official visit')} to other schools since committing"),
        (r"(\d+) unofficial visit\(s\) elsewhere since committing", lambda g: f"Took {plural(g[0], 'unofficial visit')} to other schools since committing"),
        (r"officially visited his committed school", lambda g: f"Took an official visit to {committed}"),
        (r"(\d+) higher-rated commits at his position", lambda g: f"{committed} has {g[0]} higher-rated {'commit' if g[0] == '1' else 'commits'} at his position"),
        (r"(\d+) power offer\(s\) since committing", lambda g: f"{plural(g[0], 'new Power 4 offer')} since committing"),
        (r"his own school's beat writing about a flip \((\d+) articles, (\d+) days\)", lambda g: f"{committed}'s own beat writers are discussing a possible flip ({plural(g[0], 'article')}, last {g[1]} days)"),
        (r"visited a home-state school", lambda g: "Visited a school in his home state"),
        (r"insiders predicting/favoring another school \((\d+) mentions, (\d+) days\)", lambda g: f"Insiders are predicting or leaning toward another school ({plural(g[0], 'mention')}, last {g[1]} days)"),
        (r"visited a school (\d+) miles closer to home", lambda g: f"Visited a school {g[0]} miles closer to home"),
        (r"has decommitted before", lambda g: "Has decommitted before"),
        (r"rated well above this program's usual recruit", lambda g: f"Rated well above {committed}'s typical signee"),
        (r"(\d+) On3 insider pick\(s\) for another school", lambda g: f"{plural(g[0], 'On3 insider pick')} for another school"),
        (r"flip talk in coverage \((\d+) articles, (\d+) days\)", lambda g: f"Flip talk in national coverage ({plural(g[0], 'article')}, last {g[1]} days)"),
        (r"visited elsewhere in the last (\d+) days", lambda g: f"Visited another school in the last {g[0]} days"),
        (r"(\d+) On3 insider\(s\) switched their pick away from his school", lambda g: f"{plural(g[0], 'On3 insider')} moved their pick away from {committed}"),
        (r"(\d+) visits to the same other school", lambda g: f"{g[0]} visits to the same other school"),
        (r"his school's head coach left or was fired (\d+) days ago", lambda g: f"{committed}'s head coach left or was fired {g[0]} days ago"),
        (r"decision talk alongside flip talk", lambda g: "Coverage mentions an upcoming decision alongside the flip talk"),
    ]
    out = []
    for part in (why if isinstance(why, str) else "").split(";"):
        part = part.strip()
        for pattern, say in rules:
            hit = re.search(pattern, part)
            if hit:
                out.append({"text": say(hit.groups()), "lowers": "stabilizing" in part or "officially visited his committed" in part})
                break
        else:
            if part:
                out.append({"text": part[:1].upper() + part[1:], "lowers": False})
    return sorted(out, key=lambda r: r["lowers"])


def destination_reasons(why: str) -> list[str]:
    rules = [
        (r"(\d+) official visit\(s\) since committing", lambda g: f"{plural(g[0], 'official visit')} since he committed"),
        (r"(\d+) unofficial visit\(s\) since committing", lambda g: f"{plural(g[0], 'unofficial visit')} since he committed"),
        (r"visited (\d+) days ago", lambda g: "Visited today" if g[0] == "0" else f"Last visited {plural(g[0], 'day')} ago"),
        (r"official visit before committing", lambda g: "Official visit before he committed"),
        (r"(\d+) miles closer to home", lambda g: f"{g[0]} miles closer to home"),
        (r"in-state", lambda g: "In his home state"),
        (r"was committed there before", lambda g: "He was committed there before"),
        (r"insiders favor/predict \((\d+) mentions\)", lambda g: f"Insiders lean here ({plural(g[0], 'mention')})"),
        (r"named in flip talk \((\d+) mentions\)", lambda g: f"Named in flip coverage ({plural(g[0], 'mention')})"),
        (r"(\d+) On3 insider pick\(s\)", lambda g: f"{plural(g[0], 'On3 insider pick')}"),
        (r"^offered$", lambda g: "Offered"),
    ]
    out = []
    for part in (why if isinstance(why, str) else "").split(";"):
        part = part.strip()
        for pattern, say in rules:
            hit = re.search(pattern, part)
            if hit:
                out.append(say(hit.groups()))
                break
    return out


def stability(p: float) -> str:
    return "High risk" if p >= 0.5 else "At risk" if p >= 0.35 else "Watch" if p >= 0.2 else "Solid"


def stars(rating: float) -> int | None:
    if not rating:
        return None
    return 5 if rating >= 0.9834 else 4 if rating >= 0.89 else 3 if rating >= 0.797 else 2


def num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v != v else (int(v) if v.is_integer() else round(v, 4))


def iso(value: str) -> str | None:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt).date().isoformat()
        except ValueError:
            continue
    return None


# ---- sources --------------------------------------------------------------------------------------------
def headline_titles() -> dict[str, str]:
    titles = {}
    for path in sorted(BACKFILL.glob("listings*.jsonl")) + [BACKFILL / "articles.jsonl"]:
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("url") and row.get("title"):
                titles[row["url"].rstrip("/")] = row["title"]
    return titles


def title_from_url(url: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"-\d{5,}$", "", slug)
    return slug.replace("-", " ").strip().capitalize()


def insider_note(text: str, committed: str) -> str | None:
    hit = re.search(r"insider switched away (\d+) days ago: odds x([\d.]+)", str(text or ""))
    if not hit:
        return None
    days, mult = int(hit.group(1)), float(hit.group(2))
    return (f"An On3 insider moved his pick away from {committed} {days} days ago. Past commits with a switch "
            f"{'this recent' if days < 45 else 'that old'} broke about {mult:.1f}x as often as the model's other signals alone suggest, "
            f"so his number includes that.")


def build() -> list[dict]:
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    board = pd.read_csv("data/flip_boards/live_flip_risk.csv", dtype={"player_id": str})
    dest = pd.read_csv("data/flip_boards/live_flip_destinations.csv", dtype={"player_id": str})
    rankings = pd.read_csv("data/raw_rankings.csv", dtype={"player_id": str})
    rankings = rankings[rankings["class_year"].isin(LIVE_CLASSES)].drop_duplicates("player_id")
    images = dict(pd.read_csv("data/player_images.csv", dtype={"player_id": str}).values)
    timeline = pd.read_csv("data/timeline_events.csv", dtype={"player_id": str})
    timeline = timeline[timeline["class_year"].isin(LIVE_CLASSES)]
    web = pd.read_csv("data/web_report_events.csv", dtype={"player_id": str})
    web = web[web["class_year"].isin(LIVE_CLASSES) & (web["counted"] == "yes")]
    signals = pd.read_csv("data/insider_signals.csv", dtype={"player_id": str})
    signals = signals[signals["class_year"].isin(LIVE_CLASSES)]
    m.load_on3_picks(BACKFILL)
    titles = headline_titles()

    ids = list(dict.fromkeys(list(rankings["player_id"]) + list(board["player_id"])))
    ranked = rankings.set_index("player_id")
    live = board.set_index("player_id")
    dest_by = {pid: g.sort_values("if_he_flips", ascending=False) for pid, g in dest.groupby("player_id")}
    events_by = defaultdict(list)
    for r in timeline.itertuples():
        day = iso(r.event_date)
        if day:
            events_by[r.player_id].append({"date": day, "type": r.event_type, "school": school(r.school), "source": "247Sports"})
    for r in web.itertuples():
        day = iso(r.event_date)
        if day:
            events_by[r.player_id].append({"date": day, "type": r.event_type, "school": school(r.school), "source": "Web report",
                                           "note": r.article_title if isinstance(r.article_title, str) else None,
                                           "url": r.article_url if isinstance(r.article_url, str) else None})
    coverage_by, seen_titles = defaultdict(list), defaultdict(set)
    for r in signals.sort_values("date", ascending=False).itertuples():
        if len(coverage_by[r.player_id]) >= COVERAGE_LIMIT or not isinstance(r.url, str) or not r.url:
            continue
        title = titles.get(r.url.rstrip("/")) or title_from_url(r.url)
        story = re.sub(r"\W+", " ", title.lower()).strip()[:60]
        if story in seen_titles[r.player_id]:
            continue
        seen_titles[r.player_id].add(story)
        tags = [label for flag, label in ((r.flip_talk, "Flip talk"), (r.prediction_talk, "Prediction"), (r.firm_talk, "Firm"), (r.decision_talk, "Decision"))
                if flag == 1]
        coverage_by[r.player_id].append({"date": r.date, "source": r.source, "title": title,
                                         "url": r.url, "tags": tags})

    index, board_rows, entries = [], [], []
    class_rank = board.groupby("class_year")["break_before_signing"].rank(ascending=False, method="min")
    rank_of = dict(zip(board["player_id"], class_rank))
    class_size = board.groupby("class_year").size().to_dict()
    for pid in ids:
        rk = ranked.loc[pid] if pid in ranked.index else None
        lv = live.loc[pid] if pid in live.index else None
        if isinstance(rk, pd.DataFrame):
            rk = rk.iloc[0]
        if isinstance(lv, pd.DataFrame):
            lv = lv.iloc[0]
        base = rk if rk is not None else lv
        rating = num(base.get("rating"))
        hs = str(rk["high_school"]) if rk is not None and isinstance(rk.get("high_school"), str) else ""
        hometown = re.search(r"\(([^()]*)\)\s*$", hs)
        events = sorted({(e["date"], e["type"], (e["school"] or {}).get("key")): e for e in events_by.get(pid, [])}.values(),
                        key=lambda e: e["date"], reverse=True)
        commit_events = [e for e in events if e["type"] in ("Commitment", "Decommit", "Signing", "Enrollment")]
        committed_to = school(lv["committed_school"]) if lv is not None else (
            commit_events[0]["school"] if commit_events and commit_events[0]["type"] != "Decommit" else None)
        player = {
            "id": pid, "name": str(base["name"]), "classYear": int(base["class_year"]), "position": str(base.get("position") or ""),
            "rating": rating, "stars": stars(rating or 0),
            "nationalRank": num(rk.get("national_rank")) if rk is not None else None,
            "positionRank": num(rk.get("position_rank")) if rk is not None else None,
            "stateRank": num(rk.get("state_rank")) if rk is not None else None,
            "highSchool": re.sub(r"\s*\([^()]*\)\s*$", "", hs) or None, "hometown": hometown.group(1) if hometown else None,
            "height": str(rk["height"]) if rk is not None and isinstance(rk.get("height"), str) else None,
            "weight": num(rk.get("weight")) if rk is not None else None,
            "headshot": images[pid].replace("width=70", "width=240") if pid in images else None,
            "profileUrl": str(base.get("profile_url") or "") or None,
            "committedTo": committed_to,
            "commitDate": str(lv["commitment_date"]) if lv is not None else (commit_events[0]["date"] if committed_to and commit_events else None),
        }
        flip = None
        if lv is not None:
            p, ge = float(lv["break_before_signing"]), float(lv.get("goes_elsewhere_after_break") or 0.92)
            leaves = p * ge
            name = committed_to["name"] if committed_to else "his school"
            cands = dest_by.get(pid)
            destinations, texas = [], None
            if cands is not None:
                for c in cands.itertuples():
                    entry = {"school": school(c.school_key), "ifFlips": round(float(c.if_he_flips), 4), "overall": round(float(c.if_he_flips) * leaves, 4),
                             "why": destination_reasons(c.why)}
                    destinations.append(entry)          # every school on his radar, most likely first
                    if c.school_key == "texas":
                        texas = entry
            not_recruiting = float(lv.get("not_on_radar_if_flips") or 0)
            shown = sum(d["ifFlips"] for d in destinations)
            # A player who decommits and re-commits to the same school counts as staying: every chance shown is
            # "ends up somewhere else" (the model's decommit chance x the share of decommits that don't come back).
            elsewhere = lambda x: round(float(x) * ge, 4) if x == x and x is not None else None
            flip = {
                "chance": round(leaves, 4), "low": elsewhere(lv["before_signing_low"]), "high": elsewhere(lv["before_signing_high"]),
                "chance60": elsewhere(lv["break_probability_60d"]), "label": stability(leaves), "decommitChance": round(p, 4),
                "rankInClass": int(rank_of.get(pid, 0)), "classSize": int(class_size.get(int(base["class_year"]), 0)),
                "stays": round(1 - leaves, 4), "leaves": round(leaves, 4),
                "destinations": destinations, "otherSchools": round(max(0.0, 1 - shown - not_recruiting) * leaves, 4),
                "notRecruitingYet": round(not_recruiting * leaves, 4), "texas": texas,
                "reasons": risk_reasons(lv.get("why"), name),
                "insiderNote": insider_note(lv.get("insider_switch_adjusted"), name),
                "withoutInsiderSwitch": elsewhere(lv.get("tree_before_signing")) if isinstance(lv.get("insider_switch_adjusted"), str) else None,
                "daysCommitted": num(lv.get("days_committed")), "daysToSigning": num(lv.get("days_to_signing")),
                "asOf": str(lv["snapshot_date"]),
            }
        picks = []
        for pick in m.ON3_PICKS.get(pid, []):
            if not pick["date"][:4].isdigit():
                continue
            switched = committed_to and lv is not None and m.switched_away(pick, committed_to["key"])
            picks.append({"date": pick["date"], "expert": pick["expert"], "school": school(pick["school"]),
                          "confidence": num(pick.get("confidence")), "accuracy": num(pick.get("expert_accuracy")),
                          "previousSchool": school(pick["flipped_from"]) if pick.get("flipped_from") else None,
                          "movedAway": bool(switched)})
        picks.sort(key=lambda p: p["date"], reverse=True)
        offers = [e for e in events if e["type"] == "Offer"]
        visits = [e for e in events if "Visit" in e["type"]]
        summary = {"offers": len({(e["school"] or {}).get("key") for e in offers}),
                   "officialVisits": len([e for e in visits if e["type"] == "Official Visit"]),
                   "unofficialVisits": len([e for e in visits if e["type"] == "Unofficial Visit"])}
        index.append({k: player[k] for k in ("id", "name", "classYear", "position", "stars", "nationalRank", "highSchool", "hometown")}
                     | {"committedTo": committed_to["name"] if committed_to else None,
                        "chance": flip["chance"] if flip else None, "label": flip["label"] if flip else None})
        if flip:
            board_rows.append(player | {"flip": {k: v for k, v in flip.items() if k not in ("reasons", "destinations")}
                                        | {"reasons": flip["reasons"][:3], "destinations": [d | {"why": []} for d in flip["destinations"][:3]]},
                                        "summary": summary})
        profile = player | {"flip": flip, "summary": summary, "timeline": events, "picks": picks,
                            "coverage": coverage_by.get(pid, []), "builtAt": built_at}
        entries.append({"key": f"recruiting:player:{pid}", "value": json.dumps(profile, separators=(",", ":"))})

    board_rows.sort(key=lambda r: r["flip"]["chance"], reverse=True)
    meta = {"builtAt": built_at, "asOf": str(board["snapshot_date"].max()), "classes": LIVE_CLASSES}
    entries.append({"key": "recruiting:board", "value": json.dumps(meta | {"players": board_rows}, separators=(",", ":"))})
    entries.append({"key": "recruiting:index", "value": json.dumps(meta | {"players": index}, separators=(",", ":"))})
    print(f"{len(index):,} recruits ({len(board_rows):,} live commitments), {len(entries):,} KV entries, "
          f"{sum(len(e['value']) for e in entries) / 1e6:.1f} MB")
    return entries


def upload(entries: list[dict], mode: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    batches, size = [[]], 0
    for entry in entries:
        if size + len(entry["value"]) > MAX_BATCH_BYTES and batches[-1]:
            batches.append([])
            size = 0
        batches[-1].append(entry)
        size += len(entry["value"])
    failed = 0
    for i, batch in enumerate(batches, 1):
        path = (OUT / f"recruiting-kv-{i}.json").resolve()
        path.write_text(json.dumps(batch))
        if mode == "dry-run":
            print(f"wrote {path}")
            continue
        where = ["--remote"] if mode == "remote" else ["--local", "--persist-to", ".wrangler/state"]
        result = subprocess.run(["npx", "wrangler", "kv", "bulk", "put", str(path), "--namespace-id", NAMESPACE_ID, *where], cwd=TARS)
        failed += result.returncode != 0
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--local", action="store_true", help="load into Tars's local dev KV")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return upload(build(), "dry-run" if args.dry_run else "local" if args.local else "remote")


if __name__ == "__main__":
    raise SystemExit(main())
