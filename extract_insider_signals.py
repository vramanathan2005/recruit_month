#!/usr/bin/env python3
"""Insider signals from recruiting coverage: flip talk, firm-commitment talk, schools pushing, predictions.

Reads the backfilled On3/Rivals coverage (Tars/reports/web_report/backfill: listing headlines + lead
sentences with tagged players, and full articles where they were opened) and the daily web reports (Tars'
.prebuild/reports-text.json.gz), and writes one row per (article, prospect) with rule-based signals — no
paid API. The flip models turn these into dated features ("flip talk about him in the last 30 days",
"insiders favor LSU") and test them on held-out classes like everything else.

    python3 extract_insider_signals.py        → data/insider_signals.csv (local: derived from paywalled text)
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path

from score_commitment_strength import norm_team

TARS = Path.home() / "Tars"
BACKFILL = TARS / "reports" / "web_report" / "backfill"
REPORTS = TARS / ".prebuild" / "reports-text.json.gz"
OUT = Path("data/insider_signals.csv")

FLIP = re.compile(r"\b(flip(s|ped|ping)?|flip watch|decommit\w*|re-?open\w* (his|the) recruitment|back on the market|"
                  r"still (listening|open|looking)|keeping (his|the) options open|not (fully|completely|100[- ]percent) (locked|sold|committed)|"
                  r"wavering|shaky|late push|making (a|another) (run|push)|shoot(s|ing)? (its|their) shot|gaining (ground|momentum|steam)|"
                  r"trending|in play|momentum|pursuing|push(es|ing)? (hard )?for|flip target|could (flip|move)|visit(ed|ing)? (\w+ )?(again|elsewhere))\b", re.I)
FIRM = re.compile(r"\b(reaffirm\w*|locked in|lock(s|ed)? (it )?(up|down)|shut(s|ting)? (it|things|his recruitment) down|remains? (firmly )?committed|"
                  r"solid( commit\w*)?|firm(ly)? (committed|in)|holds? (on|onto)|hang(s|ing)? on to|not going anywhere|all[- ]in|"
                  r"signs?|signed|inks?|early enroll\w*|doubl(es|ed|ing) down)\b", re.I)
# Clear prediction phrases only ("leader" or "favorite" on their own too often mean something else).
PREDICT = re.compile(r"\b(prediction machine|rpm|crystal ball|futurecast|predict\w*|smart money|(team|school) to beat|front-?runner|"
                     r"favou?rite to (land|flip|win|get)|expected to (flip|land|choose|pick|commit)|trending (toward|towards|to)|"
                     r"lean(s|ing)? (toward|towards|to)|in the driver'?s seat|pulling (it|the flip) off)\b", re.I)
# A decision coming soon — alongside flip talk, the difference between noise and a real flip.
DECISION = re.compile(r"\b(close to a decision|(nearing|nears) (a|his) (final )?decision|decision (is )?(coming|soon|near|imminent|date)|"
                      r"(set|plans?|expected) to (announce|decide|make)|announce\w*|final (visit|decision)|down to (two|three|\d)|"
                      r"narrow\w* (it|his list|things)|commitment date|decision day|in the next (few )?(days|weeks))\b", re.I)
REGISTRY = TARS / "reports" / "web_report" / "config" / "source_registry.csv"


def school_patterns() -> tuple[re.Pattern, dict[str, str]]:
    """School names as they appear in coverage → the model's school name (norm_team of 247's timeline name).
    Longest first, so "Texas A&M" and "Texas Tech" are claimed before "Texas"."""
    from fetch_coaching_changes import school_key, timeline_school_keys
    timeline = list(csv.DictReader(Path("data/timeline_events.csv").open(encoding="utf-8")))
    long_to_key = timeline_school_keys(Path("data/commit_dates.csv"), timeline)
    counts = Counter(norm_team(e["school"]) for e in timeline if e["event_type"] in ("Offer", "Commitment") and e["school"])
    model_name: dict[str, str] = {}
    for long_name, _ in counts.most_common():
        model_name.setdefault(long_to_key.get(long_name, school_key(long_name)), long_name)
    names: dict[str, str] = {}
    for raw in {e["school"] for e in timeline if e["school"] and e["event_type"] in ("Offer", "Commitment")}:
        full = re.sub(r"\s*\(.*$", "", raw).strip()
        target = model_name.get(long_to_key.get(norm_team(raw), school_key(raw)), norm_team(raw))
        names[full.lower()] = target                              # "Florida Gators", "LSU Tigers"
    for short in {r["committed_team"] for r in csv.DictReader(Path("data/commit_dates.csv").open(encoding="utf-8")) if r.get("committed_team")}:
        key = school_key(short)
        if key in model_name and len(short) >= 3:
            names.setdefault(short.lower(), model_name[key])      # "LSU", "Florida", "Texas A&M"
    # One alternation, longest names first: at each position the regex takes the first (longest) name that
    # matches, so "Texas A&M" wins over "Texas" — one pass per article instead of ~1,500.
    ordered = sorted(names, key=len, reverse=True)
    combined = re.compile(r"(?<![\w&])(" + "|".join(re.escape(n) for n in ordered) + r")(?![\w&])", re.I)
    return combined, names


def schools_in(text: str, patterns) -> list[str]:
    combined, names = patterns
    found = []
    for m in combined.finditer(text):
        target = names[m.group(1).lower()]
        if target not in found:
            found.append(target)
    return found


def site_schools(names: dict[str, str]) -> dict[str, str]:
    """Site name → the school whose beat it covers (as the model names it), or "national"."""
    out = {"247Sports national": "national"}
    for row in csv.DictReader(REGISTRY.open(encoding="utf-8")):
        school = row["school"].strip()
        out[row["source_name"]] = "national" if school.lower() == "national" else names.get(school.lower(), school.lower())
    return out


def prediction_schools(text: str, patterns) -> list[str]:
    """Schools named in the same sentence as a prediction phrase."""
    found = []
    for sentence in re.split(r"(?<=[.!?”\"])\s+", text):
        if PREDICT.search(sentence):
            for school in schools_in(sentence, patterns):
                if school not in found:
                    found.append(school)
    return found


def norm_name(value: str) -> str:
    value = re.sub(r"[^a-z ]", " ", (value or "").lower().replace("'", "").replace("’", ""))
    return " ".join(w for w in value.split() if w not in {"jr", "sr", "ii", "iii", "iv"})


def sentences_about(paragraphs: list[str], name: str) -> str:
    last = name.split()[-1]
    keep = [s for p in paragraphs for s in re.split(r"(?<=[.!?”\"])\s+", p) if name.lower() in s.lower() or re.search(rf"\b{re.escape(last)}\b", s)]
    return " ".join(keep)[:3000]


def main() -> int:
    patterns = school_patterns()
    sites = site_schools(patterns[1])
    players = {}
    for r in csv.DictReader(Path("data/raw_rankings.csv").open(encoding="utf-8")):
        players[(r["class_year"], norm_name(r["name"]))] = r["player_id"]
    by_name: dict[str, list[tuple[str, str]]] = {}
    for (year, name), pid in players.items():
        by_name.setdefault(name, []).append((year, pid))

    rows, sources = [], Counter()

    def emit(date: str, source: str, kind: str, name: str, class_year: str | None, text: str, url: str) -> None:
        key = norm_name(name)
        matches = [(class_year, players[(class_year, key)])] if class_year and (class_year, key) in players else by_name.get(key, [])
        if len(matches) != 1 or not text:
            return
        year, pid = matches[0]
        flip, firm, predict = bool(FLIP.search(text)), bool(FIRM.search(text)), bool(PREDICT.search(text))
        schools = schools_in(text, patterns)
        rows.append({"date": date, "class_year": year, "player_id": pid, "name": name, "source": source, "kind": kind,
                     "source_school": sites.get(source, "unknown"),
                     "flip_talk": int(flip), "firm_talk": int(firm), "prediction_talk": int(predict),
                     "decision_talk": int(bool(DECISION.search(text))),
                     "schools": "|".join(schools), "prediction_schools": "|".join(prediction_schools(text, patterns) if predict else []),
                     "url": url})
        sources[kind] += 1

    for path in sorted(BACKFILL.glob("listings*.jsonl")):
        for line in path.open(encoding="utf-8"):
            a = json.loads(line)
            text = f"{a['title']}. {a.get('lead', '')}"
            for p in a.get("players", []):
                emit(a["date"], a.get("source", ""), "headline", p["name"], str(p["class_year"]), text, a["url"])
    for path in sorted(BACKFILL.glob("articles*.jsonl")):
        for line in path.open(encoding="utf-8"):
            a = json.loads(line)
            for name in a.get("prospects", []):
                emit(a["date"], a.get("source", ""), "article", name, None, sentences_about([a["title"], *a["paragraphs"]], name), a["url"])
    if REPORTS.exists():
        for a in json.loads(gzip.decompress(REPORTS.read_bytes()))["articles"]:
            if a.get("kind") != "web":
                continue
            for name in a.get("prospects") or []:
                emit(a["date"], a.get("site", ""), "web report", name, None, sentences_about([a.get("title", ""), a.get("text", "")], name), a.get("url", ""))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["date"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows):,} (article, prospect) signals from {dict(sources)}; "
          f"flip talk {sum(r['flip_talk'] for r in rows):,}, firm talk {sum(r['firm_talk'] for r in rows):,}, "
          f"prediction talk {sum(r['prediction_talk'] for r in rows):,}, decision talk {sum(r['decision_talk'] for r in rows):,}; "
          f"sources: {Counter(r['source_school'] if r['source_school'] in ('national', 'unknown') else 'team site' for r in rows)} → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
