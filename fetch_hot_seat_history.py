#!/usr/bin/env python3
"""Past Coaches Hot Seat rankings (coacheshotseat.com), for testing hot seat as a flip-model signal.

The site's live table only holds the current week, so history comes from its weekly ranking posts, which
rank about the top 10 in the article ("#2. Mike Locksley – Maryland (4-8, 1-8 Big Ten)"). Posts are
fetched once at the site's robots.txt pace (10 seconds apart) and kept in data/cache/hot_seat_posts.

    python3 fetch_hot_seat_history.py        → data/hot_seat_history.csv (date, rank, coach, school, school_key, url)

Tars also saves a dated copy of the full live table every week from now on (hotseat:<date> in its KV).
"""

from __future__ import annotations

import csv
import hashlib
import html
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from score_commitment_strength import norm_team

SITEMAPS = ["https://coacheshotseat.com/post-sitemap1.xml", "https://coacheshotseat.com/post-sitemap2.xml"]
RANKING_POST = re.compile(r"hot-seat-rank|rankings-week|week-\d+-rank|hot-seat-week|preseason.*hot-seat|hot-seat-preseason|end-of-season")
CACHE = Path("data/cache/hot_seat_posts")
OUT = Path("data/hot_seat_history.csv")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
CRAWL_DELAY = 10
# A ranked line: "#2. Mike Locksley – Maryland (4-8…", "1. Mike Norvell (Florida State)", "1. Mike Norvell, Florida
# State (ACC)", or just the coach ("1. Luke Fickell and the 86 seconds"; his school is looked up by name).
RANKED = re.compile(r"^(?:#|No\.\s*)?(\d{1,2})[.):]\s+((?:[A-Z][A-Za-z'’.\-]+\.?\s){1,2}[A-Z][A-Za-z'’\-]+(?:\s(?:Jr|Sr|II|III)\.?)?)(.*)$")
SCHOOL_AFTER = [re.compile(r"^\s*\(([^)]+)\)"), re.compile(r"^\s*[–—,-]\s*([A-Z][A-Za-z.&'’ \-]+?)(?=\s*(?:\(|$|[–—:]|\s[A-Z][a-z]+\s[a-z]))")]
_last_fetch = 0.0


def fetch(url: str) -> str:
    global _last_fetch
    path = CACHE / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".html")
    if path.exists():
        return path.read_text(encoding="utf-8")
    wait = CRAWL_DELAY - (time.time() - _last_fetch)
    if wait > 0:
        time.sleep(wait)
    body = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read().decode("utf-8", "ignore")
    _last_fetch = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return body


def article_lines(page: str) -> list[str]:
    """The post body's text lines only (not the related-post excerpts after it); a bare "1." joins the next line."""
    start = page.find('class="entry-content"')
    start = start if start >= 0 else 0
    ends = [i for i in (page.find(marker, start) for marker in ('class="related', 'id="comments"', "<footer", 'class="post-navigation"', 'class="pum')) if i > 0]
    raw = re.sub(r"<(script|style).*?</\1>", " ", page[start:min(ends) if ends else len(page)], flags=re.S)
    lines = [" ".join(l.split()) for l in html.unescape(re.sub(r"<[^>]+>", "\n", raw)).split("\n")]
    lines = [l for l in lines if l]
    joined: list[str] = []
    for line in lines:
        if joined and re.fullmatch(r"#?\d{1,2}[.):]", joined[-1]):
            joined[-1] = f"{joined[-1]} {line}"
        else:
            joined.append(line)
    return joined


def post_date(page: str) -> str | None:
    hit = re.search(r'article:published_time" content="(\d{4}-\d{2}-\d{2})', page)
    return hit.group(1) if hit else None


def entries(lines: list[str]) -> list[tuple[int, str, str | None]]:
    """Ranked (rank, coach, school or None) lines, first mention of each rank."""
    found: dict[int, tuple[str, str | None]] = {}
    for line in lines:
        m = RANKED.match(line)
        if not m or not 1 <= int(m.group(1)) <= 30 or int(m.group(1)) in found:
            continue
        school = next((hit.group(1).strip() for pattern in SCHOOL_AFTER if (hit := pattern.match(m.group(3)))), None)
        found[int(m.group(1))] = (m.group(2).strip(), school)
    return [(r, *found[r]) for r in sorted(found)]


def coach_schools() -> dict[str, str]:
    """Coach name → school, for lines that only name the coach: today's table, then past coaching changes."""
    known: dict[str, str] = {}
    live = Path("data/cache/hot_seat_current.json")
    if live.exists():
        import json
        for c in json.loads(live.read_text())["coaches"]:
            known[c["coach"].lower()] = c["school"]
    for r in csv.DictReader(Path("data/coaching_changes.csv").open(encoding="utf-8")):
        known.setdefault(r["outgoing_coach"].lower(), r["school"])
    return known


def main() -> int:
    urls: list[str] = []
    for sitemap in SITEMAPS:
        urls += [u for u in re.findall(r"<loc>([^<]+)</loc>", fetch(sitemap)) if RANKING_POST.search(u) and "conference" not in u]   # conference-only lists rank within one league
    rows, parsed = [], []
    for url in urls:
        page = fetch(url)
        parsed.append((url, post_date(page), entries(article_lines(page))))
    # A coach named without a school takes the school he's listed with elsewhere in these posts, else today's table.
    known = coach_schools()
    for _, _, ranked in parsed:
        for _, coach, school in ranked:
            if school:
                known[coach.lower()] = school
    for url, day, ranked in sorted(parsed, key=lambda p: p[1] or ""):
        ranked = [(r, coach, school or known.get(coach.lower())) for r, coach, school in ranked]
        usable = [x for x in ranked if x[2]]
        print(f"{day or '????-??-??'}  {len(usable):2d} ranked  {url.rsplit('/', 2)[-2][:70]}")
        if len(usable) < 5 or not day:          # not a ranking post (a reader poll, a single-coach story)
            continue
        rows += [{"date": day, "rank": r, "coach": coach, "school": school, "school_key": norm_team(school), "url": url}
                 for r, coach, school in usable]
    rows.sort(key=lambda r: (r["date"], r["rank"]))
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "rank", "coach", "school", "school_key", "url"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} ranked entries from {len({r['url'] for r in rows})} posts → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
