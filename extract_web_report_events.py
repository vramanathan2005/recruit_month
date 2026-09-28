#!/usr/bin/env python3
"""Extract recruiting visits and offers from the Tars web report articles.

The daily web reports often have a visit days before 247's timeline does ("details game day visit to
Florida"). This reads each web report article with OpenAI, pulls out events that already happened, matches
the prospect to 247's rankings, drops anything 247's timeline already has, and writes the rest to
data/web_report_events.csv. train_flip_snapshot_model.py adds those rows to the timeline, so they feed
the same visit/offer features the model learned from 247 — no new, untrained signals.

Articles come from Tars (`npm run sync:reports` writes .prebuild/reports-text.json.gz). Each article's
extraction is cached in data/cache/web_report_extractions.jsonl, so a re-run only pays for new articles.

    python3 extract_web_report_events.py [--limit 20] [--only "Easton Royal"]

Needs OPENAI_API_KEY in the environment or in Tars' .dev.vars.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests

from score_commitment_strength import norm_team, parse_date

TARS = Path.home() / "Tars"
MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-luna")
# Only visits and offers. Commitments and decommits define the outcome the model predicts, and articles
# restate them constantly ("remains committed", "among the top pledges") — 247 logs them within a day anyway.
EVENT_TYPES = ["Official Visit", "Unofficial Visit", "Offer"]
LIVE_CLASSES = {"2026", "2027", "2028"}
SAME_EVENT_DAYS = 7

INSTRUCTIONS = """You extract campus visits and new offers for HIGH SCHOOL football prospects from one news article.

Return every such event the article reports for a high school prospect (not college players, transfers or coaches):
- "Official Visit": the prospect took an official visit to a college.
- "Unofficial Visit": the prospect visited a campus otherwise — game day visits, junior days, camps, spring
  practice, "was on campus". Use Official only when the article says official.
- "Offer": a college newly offered the prospect, reported as news in this article (not a list of old offers).

For each event:
- player_name: the prospect's full name as written.
- school: the college's common short name, e.g. "Florida", "Texas A&M", "Ohio State", "LSU", "Ole Miss".
- event_date: the date the event itself happened (YYYY-MM-DD), worked out from the article date and phrases
  like "Saturday" or "last weekend". Articles often recap older visits ("gave LSU an official visit in June"):
  give the real date if the article states it, otherwise leave it empty. Never use the article date for an
  event the article doesn't place on that day.
- date_is_approximate: true unless the article pins the event to a specific day.
- status: "completed" if it already happened by the article date; "planned" if it is scheduled or expected.
- evidence: a short quote (under 25 words) from the article that shows the event.

Only report events the article states. Do not infer visits from speculation ("could visit", "hopes to get
him on campus" are not events). Return an empty list when there are none."""

SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string"},
                    "event_type": {"type": "string", "enum": EVENT_TYPES},
                    "school": {"type": "string"},
                    "event_date": {"type": "string"},
                    "date_is_approximate": {"type": "boolean"},
                    "status": {"type": "string", "enum": ["completed", "planned"]},
                    "evidence": {"type": "string"},
                },
                "required": ["player_name", "event_type", "school", "event_date", "date_is_approximate", "status", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["events"],
    "additionalProperties": False,
}

OUT_FIELDS = [
    "class_year", "player_id", "name", "event_type", "event_date", "school", "status", "date_is_approximate",
    "counted", "why_not_counted", "article_date", "article_title", "article_url", "evidence",
]


def api_key(env_file: Path) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit(f"OPENAI_API_KEY isn't set (looked in the environment and {env_file}).")
    return key


def article_key(article: dict) -> str:
    return f"{article.get('date', '')}|{article.get('url') or article.get('title', '')}"


def extract(article: dict, key: str) -> list[dict]:
    text = article.get("text", "")[:14000]
    prospects = ", ".join(article.get("prospects") or [])
    body = {
        "model": MODEL,
        "instructions": INSTRUCTIONS,
        "input": f"Article date: {article['date']}\nTitle: {article.get('title', '')}\nSite: {article.get('site', '')}\n"
                 + (f"Prospects this article is about: {prospects}\n" if prospects else "")
                 + f"\n{text}",
        "max_output_tokens": 4000,
        "reasoning": {"effort": "low"},
        "store": False,
        "text": {"format": {"type": "json_schema", "name": "events", "schema": SCHEMA, "strict": True}},
    }
    for attempt in range(4):
        try:
            response = requests.post("https://api.openai.com/v1/responses", json=body, timeout=120,
                                     headers={"authorization": f"Bearer {key}", "content-type": "application/json"})
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                raise requests.RequestException(f"HTTP {response.status_code}")
            response.raise_for_status()
            data = response.json()
            output = next(part["text"] for item in data.get("output", []) if item.get("type") == "message"
                          for part in item.get("content", []) if part.get("type") == "output_text")
            return json.loads(output)["events"]
        except (requests.RequestException, StopIteration, json.JSONDecodeError, KeyError) as error:
            if attempt == 3:
                raise RuntimeError(f"{article.get('title', '')[:60]}: {error}") from None
            import time
            time.sleep(2 ** attempt * 2)
    return []


def norm_name(value: str) -> str:
    value = re.sub(r"[^a-z ]", " ", (value or "").lower().replace("’", "'").replace("'", ""))
    return " ".join(word for word in value.split() if word not in {"jr", "sr", "ii", "iii", "iv", "v"})


def school_matches(a: str, b: str) -> bool:
    """247 school strings are messy ("Tennessee Volunteers (March 27-28) for spring practice")."""
    x, y = norm_team(a), norm_team(b)
    return bool(x and y) and (x == y or x.startswith(y + " ") or y.startswith(x + " ") or x.startswith(y + " (") or y.startswith(x + " ("))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reports", type=Path, default=TARS / ".prebuild" / "reports-text.json.gz")
    parser.add_argument("--env-file", type=Path, default=TARS / ".dev.vars")
    parser.add_argument("--rankings", type=Path, default=Path("data/raw_rankings.csv"))
    parser.add_argument("--timeline", type=Path, default=Path("data/timeline_events.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/web_report_extractions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/web_report_events.csv"))
    parser.add_argument("--limit", type=int, help="only extract this many new articles (for testing)")
    parser.add_argument("--only", help="only articles mentioning this name (for testing)")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    # One entry per article (the report index can list an article once per prospect it's about).
    articles = list({article_key(a): a for a in json.loads(gzip.decompress(args.reports.read_bytes()))["articles"] if a.get("kind") == "web"}.values())
    if args.only:
        articles = [a for a in articles if args.only.lower() in (a.get("text", "") + " ".join(a.get("prospects") or [])).lower()]

    cache: dict[str, list[dict]] = {}
    if args.cache.exists():
        for line in args.cache.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[row["key"]] = row["events"]
    todo = [a for a in articles if article_key(a) not in cache]
    print(f"{len(articles)} web report articles; {len(articles) - len(todo)} already extracted", flush=True)
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"extracting {len(todo)} with {MODEL}", flush=True)

    if todo:
        key = api_key(args.env_file)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        failures = 0
        with args.cache.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(extract, a, key): a for a in todo}
            for done, future in enumerate(as_completed(futures), 1):
                article = futures[future]
                try:
                    events = future.result()
                except RuntimeError as error:
                    failures += 1
                    print(f"  failed: {error}", flush=True)
                    continue
                cache[article_key(article)] = events
                handle.write(json.dumps({"key": article_key(article), "events": events}) + "\n")
                handle.flush()
                if done % 50 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)} articles", flush=True)
        if failures:
            print(f"WARNING: {failures} articles failed; they'll be retried on the next run", flush=True)

    # Match prospects to 247 (live classes only), and index 247's own timeline for de-duplication.
    by_name: dict[str, list[dict]] = {}
    for row in csv.DictReader(args.rankings.open(encoding="utf-8")):
        if row["class_year"] in LIVE_CLASSES:
            by_name.setdefault(norm_name(row["name"]), []).append(row)
    known: dict[tuple[str, str], list[tuple[date, str]]] = {}
    for row in csv.DictReader(args.timeline.open(encoding="utf-8")):
        day = parse_date(row["event_date"])
        if day:
            known.setdefault((row["player_id"], row["event_type"]), []).append((day, row["school"]))

    rows: list[dict] = []
    for article in sorted(articles, key=lambda a: a["date"]):
        article_day = date.fromisoformat(article["date"])
        for event in cache.get(article_key(article), []):
            matches = {r["player_id"]: r for r in by_name.get(norm_name(event["player_name"]), [])}
            recruit = next(iter(matches.values())) if len(matches) == 1 else None
            try:
                day = date.fromisoformat(event["event_date"]) if event["event_date"] else None
            except ValueError:
                day = None
            approximate = event["date_is_approximate"] or day is None
            day = day or article_day
            why = ""
            if event["event_type"] not in EVENT_TYPES:
                why = "not a visit or offer"
            elif event["status"] != "completed":
                why = "planned, not yet happened"
            elif approximate:
                why = "date not stated exactly"
            elif not matches:
                why = "prospect not in 247's 2026-2028 rankings"
            elif len(matches) > 1:
                why = "more than one ranked prospect with this name"
            elif day > article_day + timedelta(days=1):
                why = "dated after the article"
            elif any(abs((day - known_day).days) <= SAME_EVENT_DAYS and school_matches(event["school"], school)
                     for known_day, school in known.get((recruit["player_id"], event["event_type"]), [])):
                why = "already in 247's timeline"
            rows.append({
                "class_year": recruit["class_year"] if recruit else "",
                "player_id": recruit["player_id"] if recruit else "",
                "name": recruit["name"] if recruit else event["player_name"],
                "event_type": event["event_type"],
                "event_date": f"{day.month}/{day.day}/{day.year}",
                "school": event["school"],
                "status": event["status"],
                "date_is_approximate": "yes" if approximate else "no",
                "counted": "no" if why else "yes",
                "why_not_counted": why,
                "article_date": article["date"],
                "article_title": article.get("title", ""),
                "article_url": article.get("url", ""),
                "evidence": event["evidence"],
            })

    # The same visit is often reported in several articles: count it once (the first report).
    counted: list[dict] = []
    for row in rows:
        if row["counted"] != "yes":
            continue
        day = parse_date(row["event_date"])
        if any(r["player_id"] == row["player_id"] and r["event_type"] == row["event_type"] and school_matches(r["school"], row["school"])
               and abs((parse_date(r["event_date"]) - day).days) <= SAME_EVENT_DAYS for r in counted):
            row["counted"], row["why_not_counted"] = "no", "same event reported in an earlier article"
        else:
            counted.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    by_type: dict[str, int] = {}
    for row in counted:
        by_type[row["event_type"]] = by_type.get(row["event_type"], 0) + 1
    reasons: dict[str, int] = {}
    for row in rows:
        if row["why_not_counted"]:
            reasons[row["why_not_counted"]] = reasons.get(row["why_not_counted"], 0) + 1
    print(f"{len(rows)} events extracted; {len(counted)} new to 247 and counted: {by_type}")
    print(f"not counted: {reasons}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
