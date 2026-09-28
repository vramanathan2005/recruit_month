#!/usr/bin/env python3
"""Scrape 247Sports high-school recruit commitment dates for 2022-2027.

The player pool comes from CompositeRecruitRankings with
InstitutionGroup=HighSchool. Commitment dates come from the matching
class-level Football Commits pages, which expose "Committed: M/D/YYYY" and
avoid current NCAA/transfer state on living player profiles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE = "https://247sports.com"
YEARS = (2022, 2023, 2024, 2025, 2026, 2027, 2028)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    )
}
PLAYER_ID_RE = re.compile(r"/player/[^/]+-(\d+)", re.IGNORECASE)
COMMIT_DATE_RE = re.compile(r"Committed:\s*(\d{1,2}/\d{1,2}/\d{4})")
TIMELINE_HEADER_RE = re.compile(
    r"^(\w+ \d{1,2}, \d{4}):\s*(Commitment|Decommit|Signing|Enrollment|Offer|Official Visit|Unofficial Visit)$"
)


@dataclass
class RankingRow:
    class_year: int
    player_id: str
    name: str
    profile_url: str
    high_school: str
    position: str
    height: str
    weight: str
    rating: str
    primary_rank: str
    other_rank: str
    national_rank: str
    position_rank: str
    state_rank: str
    ranking_team: str
    recruitment_url: str


@dataclass
class CommitRow:
    class_year: int
    player_id: str
    name: str
    profile_url: str
    high_school: str
    position: str
    height: str
    weight: str
    rating: str
    national_rank: str
    position_rank: str
    state_rank: str
    committed_team: str
    committed_date: str


@dataclass
class RecruitmentDetailRow:
    class_year: int
    player_id: str
    name: str
    recruitment_url: str
    committed_team: str
    commitment_type: str
    committed_date: str


def norm_space(text: str) -> str:
    return " ".join(text.split())


def text_or_empty(node) -> str:
    return norm_space(node.get_text(" ", strip=True)) if node else ""


def player_id_from_url(url: str) -> str:
    match = PLAYER_ID_RE.search(url)
    return match.group(1) if match else ""


def parse_date(value: str) -> date | None:
    if not value or value.upper() == "NA":
        return None
    try:
        return datetime.strptime(value, "%m/%d/%Y").date()
    except ValueError:
        return None


class Fetcher:
    def __init__(self, cache_dir: Path, delay: float, refresh: bool = False, refresh_years: set[int] | None = None):
        self.cache_dir = cache_dir
        self.delay = delay
        self.refresh = refresh
        # Classes still being recruited: their pages are re-downloaded; finished classes reuse the cache.
        self.refresh_years = refresh_years or set()
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.last_fetch = 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def refresh_for(self, year: int) -> bool:
        return self.refresh or year in self.refresh_years

    def get(self, url: str, params: dict[str, str] | None = None, refresh: bool | None = None) -> str:
        params = params or {}
        key = url + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        cache_file = self.cache_dir / (hashlib.sha1(key.encode("utf-8")).hexdigest() + ".html")
        if cache_file.exists() and not (self.refresh if refresh is None else refresh):
            return cache_file.read_text(encoding="utf-8")

        elapsed = time.time() - self.last_fetch
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        last_error: requests.RequestException | None = None
        for attempt in range(4):
            try:
                response = self.session.get(url, params=params, timeout=(10, 60))
                self.last_fetch = time.time()
                response.raise_for_status()
                cache_file.write_text(response.text, encoding="utf-8")
                return response.text
            except requests.RequestException as error:
                last_error = error
                if attempt == 3:
                    break
                time.sleep(2 ** attempt)
        raise last_error or requests.RequestException(f"Failed to fetch {url}")


def cache_path_for(cache_dir: Path, url: str, params: dict[str, str] | None = None) -> Path:
    params = params or {}
    key = url + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return cache_dir / (hashlib.sha1(key.encode("utf-8")).hexdigest() + ".html")


def fetch_detail_html(cache_dir: Path, url: str, refresh: bool = False) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path_for(cache_dir, url)
    if cache_file.exists() and not refresh:
        return cache_file.read_text(encoding="utf-8")
    last_error: requests.RequestException | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, headers=HEADERS, timeout=(10, 60))
            response.raise_for_status()
            cache_file.write_text(response.text, encoding="utf-8")
            return response.text
        except requests.RequestException as error:
            last_error = error
            if attempt == 3:
                break
            time.sleep(2 ** attempt)
    raise last_error or requests.RequestException(f"Failed to fetch {url}")


def first_attr(nodes: Iterable, attr: str) -> str:
    for node in nodes:
        value = node.get(attr)
        if value:
            return norm_space(value)
    return ""


def parse_rankings_page(html: str, class_year: int) -> list[RankingRow]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[RankingRow] = []
    for item in soup.select("li.rankings-page__list-item"):
        if item.get_text(" ", strip=True).lower() == "load more":
            continue

        recruit = item.select_one("div.recruit")
        name_link = recruit.find("a", href=re.compile(r"/player/", re.I)) if recruit else None
        if not name_link:
            continue

        profile_url = urljoin(BASE, name_link.get("href", ""))
        player_id = player_id_from_url(profile_url)
        if not player_id:
            continue

        metrics = text_or_empty(item.select_one("div.metrics"))
        height, weight = "", ""
        if "/" in metrics:
            height, weight = [part.strip() for part in metrics.split("/", 1)]

        rank_col = item.select_one("div.rank-column")
        recruitment_link = item.find("a", href=re.compile(r"/recruitment/", re.I))
        rows.append(
            RankingRow(
                class_year=class_year,
                player_id=player_id,
                name=text_or_empty(name_link),
                profile_url=profile_url,
                high_school=text_or_empty(recruit.select_one("span.meta") if recruit else None),
                position=text_or_empty(item.select_one("div.position")),
                height=height,
                weight=weight,
                rating=text_or_empty(item.select_one("span.score")),
                primary_rank=text_or_empty(rank_col.select_one("div.primary") if rank_col else None),
                other_rank=text_or_empty(rank_col.select_one("div.other") if rank_col else None),
                national_rank=text_or_empty(item.select_one("a.natrank")),
                position_rank=text_or_empty(item.select_one("a.posrank")),
                state_rank=text_or_empty(item.select_one("a.sttrank")),
                ranking_team=first_attr(item.select("div.status img, div.status a img"), "title"),
                recruitment_url=urljoin(BASE, recruitment_link.get("href", "")) if recruitment_link else "",
            )
        )
    return rows


def parse_commits_page(html: str, class_year: int) -> list[CommitRow]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[CommitRow] = []
    for item in soup.select("li.ri-page__list-item, li.commitment-list__item"):
        text = item.get_text(" ", strip=True)
        date_match = COMMIT_DATE_RE.search(text)
        name_link = item.find("a", href=re.compile(r"/player/", re.I))
        if not date_match or not name_link:
            continue

        profile_url = urljoin(BASE, name_link.get("href", ""))
        player_id = player_id_from_url(profile_url)
        if not player_id:
            continue

        metrics = text_or_empty(item.select_one("div.metrics"))
        height, weight = "", ""
        if "/" in metrics:
            height, weight = [part.strip() for part in metrics.split("/", 1)]

        school_node = item.select_one("span.meta")
        team = first_attr(item.select("div.status img, div.commitment-list__status img, img"), "title")
        if team == text_or_empty(name_link):
            team = ""

        rows.append(
            CommitRow(
                class_year=class_year,
                player_id=player_id,
                name=text_or_empty(name_link),
                profile_url=profile_url,
                high_school=text_or_empty(school_node),
                position=text_or_empty(item.select_one("div.position")),
                height=height,
                weight=weight,
                rating=text_or_empty(item.select_one("span.score")),
                national_rank=text_or_empty(item.select_one("a.natrank")),
                position_rank=text_or_empty(item.select_one("a.posrank")),
                state_rank=text_or_empty(item.select_one("a.sttrank")),
                committed_team=team,
                committed_date=date_match.group(1),
            )
        )
    return rows


def parse_recruitment_detail(html: str, recruit: RankingRow) -> RecruitmentDetailRow | None:
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", id="predictionList")
    if not script:
        return None

    try:
        data = json.loads(script.string or script.get_text() or "[]")
    except json.JSONDecodeError:
        return None

    if not data:
        return None
    block = (data[0].get("commitmentBlock") or [{}])[0]
    committed_date = block.get("date") or ""
    if not committed_date or committed_date in {"1/1/0001", "NA"}:
        return None

    return RecruitmentDetailRow(
        class_year=recruit.class_year,
        player_id=recruit.player_id,
        name=recruit.name,
        recruitment_url=recruit.recruitment_url,
        committed_team=block.get("institution") or "",
        commitment_type=block.get("commitmentType") or "",
        committed_date=committed_date,
    )


@dataclass
class TimelineEventRow:
    class_year: int
    player_id: str
    name: str
    event_type: str
    event_date: str
    school: str


def parse_timeline_events(html: str, recruit: RankingRow) -> list[TimelineEventRow]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("section.main, div.main-div, div#main")
    if not main:
        return []
    rows = []

    for item in main.select("ul.timeline-event-index_lst > li"):
        header = text_or_empty(item.select_one("b"))
        match = TIMELINE_HEADER_RE.match(header)
        if not match:
            continue

        date_str, event_type = match.group(1), match.group(2)
        body = text_or_empty(item.select_one("p"))
        school = timeline_school_from_body(body, event_type, recruit.name)
        if not school:
            continue

        try:
            dt = datetime.strptime(date_str, "%B %d, %Y")
            date_fmt = f"{dt.month}/{dt.day}/{dt.year}"
        except ValueError:
            continue
        rows.append(TimelineEventRow(
            class_year=recruit.class_year,
            player_id=recruit.player_id,
            name=recruit.name,
            event_type=event_type,
            event_date=date_fmt,
            school=school,
        ))
    return rows


def timeline_school_from_body(body: str, event_type: str, player_name: str) -> str:
    body = norm_space(body)
    escaped_name = re.escape(player_name)
    patterns = {
        "Commitment": rf"{escaped_name}\s+commits to\s+(.+)$",
        "Decommit": rf"{escaped_name}\s+decommits from\s+(.+)$",
        "Signing": rf"{escaped_name}\s+signs with\s+(.+)$",
        "Enrollment": rf"{escaped_name}\s+enrolls at\s+(.+)$",
        "Offer": rf"(.+?)\s+offers?\s+{escaped_name}$",
        "Official Visit": rf"{escaped_name}\s+officially visits\s+(.+)$",
        "Unofficial Visit": rf"{escaped_name}\s+unofficially visits\s+(.+)$",
    }
    pattern = patterns.get(event_type)
    if not pattern:
        return ""
    match = re.search(pattern, body, re.IGNORECASE)
    return norm_space(match.group(1)) if match else ""


def timeline_url(profile_url: str) -> str:
    """https://247sports.com/player/<slug>-<id>/high-school-<n>/ → .../player/<slug>-<id>/timelineevents/"""
    match = re.match(r"(https?://[^/]+/player/[^/]+)/", profile_url.rstrip("/") + "/", re.I)
    return (match.group(1) if match else profile_url.rstrip("/")) + "/timelineevents/"


def scrape_timeline_events(
    fetcher: Fetcher,
    rankings: list[RankingRow],
    workers: int = 8,
) -> list[TimelineEventRow]:
    rows: list[TimelineEventRow] = []

    failed: list[str] = []

    def scrape_one(recruit: RankingRow) -> list[TimelineEventRow]:
        if not recruit.profile_url:
            return []
        # 247 moved the full timeline from .../player/<slug>-<id>/high-school-<n>/timelineevents/ (now a
        # 404) to .../player/<slug>-<id>/timelineevents/. Pages cached under the old address are still
        # used for classes that aren't being refreshed, so finished classes aren't re-downloaded.
        legacy_url = recruit.profile_url.rstrip("/") + "/timelineevents/"
        url = timeline_url(recruit.profile_url)
        refresh = fetcher.refresh_for(recruit.class_year)
        legacy_cache = cache_path_for(fetcher.cache_dir, legacy_url)
        if not refresh and legacy_cache.exists():
            return parse_timeline_events(legacy_cache.read_text(encoding="utf-8"), recruit)
        try:
            html = fetch_detail_html(fetcher.cache_dir, url, refresh)
        except requests.RequestException:
            failed.append(recruit.profile_url)
            if not legacy_cache.exists():
                return []
            html = legacy_cache.read_text(encoding="utf-8")
        return parse_timeline_events(html, recruit)

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(scrape_one, r): r for r in rankings}
        for future in as_completed(futures):
            completed += 1
            rows.extend(future.result() or [])
            if completed % 500 == 0 or completed == len(rankings):
                print(f"timeline events {completed}/{len(rankings)}: {len(rows)} events, {len(failed)} failed", flush=True)
    if failed:
        print(f"WARNING: {len(failed)} timeline pages failed to download (kept the cached copy where there was one)", flush=True)
    return rows


def scrape_recruitment_details(
    fetcher: Fetcher,
    rankings: list[RankingRow],
    limit: int | None = None,
    workers: int = 6,
) -> list[RecruitmentDetailRow]:
    rows: list[RecruitmentDetailRow] = []
    candidates = [row for row in rankings if row.recruitment_url and row.ranking_team]
    if limit is not None:
        candidates = candidates[:limit]

    def scrape_one(recruit: RankingRow) -> RecruitmentDetailRow | None:
        try:
            html = fetch_detail_html(fetcher.cache_dir, recruit.recruitment_url, fetcher.refresh_for(recruit.class_year))
            return parse_recruitment_detail(html, recruit)
        except requests.RequestException:
            return None

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(scrape_one, recruit) for recruit in candidates]
        for future in as_completed(futures):
            completed += 1
            detail = future.result()
            if detail:
                rows.append(detail)
            if completed % 100 == 0 or completed == len(candidates):
                print(f"recruitment details {completed}/{len(candidates)}: {len(rows)} rows", flush=True)
    return rows


def scrape_paged(
    fetcher: Fetcher,
    url: str,
    class_year: int,
    parser,
    label: str,
    max_pages: int | None,
    base_params: dict[str, str] | None = None,
) -> list:
    rows = []
    page = 1
    while max_pages is None or page <= max_pages:
        params = dict(base_params or {})
        params["Page"] = str(page)
        html = fetcher.get(url, params=params, refresh=fetcher.refresh_for(class_year))
        page_rows = parser(html, class_year)
        print(f"{class_year} {label} page {page}: {len(page_rows)} rows", flush=True)
        if not page_rows:
            break
        rows.extend(page_rows)
        page += 1
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dataclass_dicts(rows) -> list[dict]:
    return [row.__dict__.copy() for row in rows]


def star_bucket(rating: str) -> str:
    try:
        value = float(rating)
    except ValueError:
        return ""
    if value >= 0.98 or value >= 98:
        return "5-star"
    if value >= 0.89 or value >= 89:
        return "4-star"
    if value >= 0.80 or value >= 80:
        return "3-star"
    return "2-star/NR"


def join_rows(
    rankings: list[RankingRow],
    commits: list[CommitRow],
    details: list[RecruitmentDetailRow],
) -> list[dict]:
    commits_by_id: dict[tuple[int, str], CommitRow] = {}
    for commit in commits:
        key = (commit.class_year, commit.player_id)
        old = commits_by_id.get(key)
        if not old:
            commits_by_id[key] = commit
            continue
        old_date = parse_date(old.committed_date)
        new_date = parse_date(commit.committed_date)
        if old_date is None or (new_date is not None and new_date > old_date):
            commits_by_id[key] = commit

    details_by_id: dict[tuple[int, str], RecruitmentDetailRow] = {
        (detail.class_year, detail.player_id): detail for detail in details
    }

    joined = []
    for recruit in rankings:
        commit = commits_by_id.get((recruit.class_year, recruit.player_id))
        detail = details_by_id.get((recruit.class_year, recruit.player_id))
        committed_date = detail.committed_date if detail else (commit.committed_date if commit else "")
        committed_team = detail.committed_team if detail else (commit.committed_team if commit else recruit.ranking_team)
        dt = parse_date(committed_date)
        joined.append(
            {
                "class_year": recruit.class_year,
                "player_id": recruit.player_id,
                "name": recruit.name,
                "profile_url": recruit.profile_url,
                "high_school": recruit.high_school,
                "position": recruit.position,
                "height": recruit.height,
                "weight": recruit.weight,
                "rating": recruit.rating,
                "star_bucket": star_bucket(recruit.rating),
                "primary_rank": recruit.primary_rank,
                "other_rank": recruit.other_rank,
                "national_rank": recruit.national_rank,
                "position_rank": recruit.position_rank,
                "state_rank": recruit.state_rank,
                "committed_team": committed_team,
                "commitment_type": detail.commitment_type if detail else "",
                "committed_date": committed_date,
                "commit_month": dt.strftime("%Y-%m") if dt else "",
                "commit_week_monday": (dt.fromordinal(dt.toordinal() - dt.weekday()).isoformat() if dt else ""),
                "commit_day_of_year": dt.strftime("%m-%d") if dt else "",
                "commit_date_source": "recruitment_detail" if detail else ("commit_page" if commit else ""),
                "matched_commit_page": "yes" if commit else "no",
                "matched_recruitment_detail": "yes" if detail else "no",
            }
        )
    return joined


def summarize(joined: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    monthly = Counter()
    weekly = Counter()
    daily = Counter()
    for row in joined:
        if not row["committed_date"]:
            continue
        dt = parse_date(row["committed_date"])
        if not dt:
            continue
        monthly[(row["class_year"], dt.strftime("%Y-%m"))] += 1
        weekly[(row["class_year"], dt.fromordinal(dt.toordinal() - dt.weekday()).isoformat())] += 1
        daily[(row["class_year"], dt.isoformat())] += 1

    month_rows = [
        {"class_year": year, "month": month, "commits": count}
        for (year, month), count in sorted(monthly.items())
    ]
    week_rows = [
        {"class_year": year, "week_monday": week, "commits": count}
        for (year, week), count in sorted(weekly.items())
    ]
    day_rows = [
        {"class_year": year, "date": day, "commits": count}
        for (year, day), count in sorted(daily.items())
    ]
    return month_rows, week_rows, day_rows


def summary_notes(joined: list[dict]) -> str:
    by_year = defaultdict(list)
    for row in joined:
        by_year[row["class_year"]].append(row)

    lines = ["# Scrape Summary", ""]
    total = len(joined)
    matched = sum(1 for row in joined if row["committed_date"])
    detail_matched = sum(1 for row in joined if row["matched_recruitment_detail"] == "yes")
    lines.append(f"- Ranked high-school recruits: {total}")
    lines.append(f"- Matched to a 247 commit date: {matched} ({matched / total:.1%})" if total else "- Matched: 0")
    lines.append(f"- Matched via recruitment detail page: {detail_matched} ({detail_matched / total:.1%})" if total else "")
    lines.append("")
    lines.append("| Class | Recruits | Matched | June | July | July 4 | Aug 1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for year in sorted(by_year):
        rows = by_year[year]
        matched_rows = [row for row in rows if row["committed_date"]]
        june = july = july4 = aug1 = 0
        for row in matched_rows:
            dt = parse_date(row["committed_date"])
            if not dt:
                continue
            june += dt.month == 6
            july += dt.month == 7
            july4 += dt.month == 7 and dt.day == 4
            aug1 += dt.month == 8 and dt.day == 1
        lines.append(f"| {year} | {len(rows)} | {len(matched_rows)} | {june} | {july} | {july4} | {aug1} |")
    lines.append("")
    lines.append("Data definition: the recruit universe is 247Sports Composite high-school football rankings.")
    lines.append("Commitment dates prefer each player's 247Sports recruitment detail commitmentBlock.")
    lines.append("Class-level Football Commits pages are used as fallback and joined by player ID.")
    lines.append("Transfer Portal pages and current NCAA profile state are not used for commitment dates.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--refresh", action="store_true", help="Re-download every page.")
    parser.add_argument("--refresh-years", nargs="+", type=int, default=[],
                        help="Re-download only these classes' pages (the ones still being recruited).")
    parser.add_argument("--max-pages", type=int, default=None, help="Debug limit per source/year.")
    parser.add_argument("--skip-recruitment-details", action="store_true")
    parser.add_argument("--skip-commit-pages", action="store_true")
    parser.add_argument("--skip-timeline-events", action="store_true")
    parser.add_argument("--detail-limit", type=int, default=None, help="Debug limit for recruitment detail pages.")
    parser.add_argument("--detail-workers", type=int, default=6)
    parser.add_argument("--timeline-workers", type=int, default=8)
    args = parser.parse_args()

    fetcher = Fetcher(args.cache_dir, args.delay, args.refresh, set(args.refresh_years))
    rankings: list[RankingRow] = []
    commits: list[CommitRow] = []
    details: list[RecruitmentDetailRow] = []

    for year in args.years:
        ranking_url = f"{BASE}/Season/{year}-Football/CompositeRecruitRankings/"
        commit_url = f"{BASE}/Season/{year}-Football/Commits/"
        rankings.extend(
            scrape_paged(
                fetcher,
                ranking_url,
                year,
                parse_rankings_page,
                "rankings",
                args.max_pages,
                {"InstitutionGroup": "HighSchool"},
            )
        )
        if not args.skip_commit_pages:
            commits.extend(scrape_paged(fetcher, commit_url, year, parse_commits_page, "commits", args.max_pages))

    if not args.skip_recruitment_details:
        details = scrape_recruitment_details(fetcher, rankings, args.detail_limit, args.detail_workers)

    timeline_events: list[TimelineEventRow] = []
    if not args.skip_timeline_events:
        timeline_events = scrape_timeline_events(fetcher, rankings, args.timeline_workers)

    joined = join_rows(rankings, commits, details)
    month_rows, week_rows, day_rows = summarize(joined)

    if timeline_events:
        write_csv(
            args.out_dir / "timeline_events.csv",
            dataclass_dicts(timeline_events),
            list(TimelineEventRow.__annotations__),
        )
        print(f"Wrote {len(timeline_events)} timeline events to {args.out_dir / 'timeline_events.csv'}", flush=True)

    write_csv(args.out_dir / "raw_rankings.csv", dataclass_dicts(rankings), list(RankingRow.__annotations__))
    write_csv(args.out_dir / "raw_commits.csv", dataclass_dicts(commits), list(CommitRow.__annotations__))
    write_csv(
        args.out_dir / "raw_recruitment_details.csv",
        dataclass_dicts(details),
        list(RecruitmentDetailRow.__annotations__),
    )
    write_csv(args.out_dir / "commit_dates.csv", joined, list(joined[0].keys()) if joined else [])
    write_csv(args.out_dir / "summary_by_month.csv", month_rows, ["class_year", "month", "commits"])
    write_csv(args.out_dir / "summary_by_week.csv", week_rows, ["class_year", "week_monday", "commits"])
    write_csv(args.out_dir / "summary_by_day.csv", day_rows, ["class_year", "date", "commits"])
    (args.out_dir / "summary.md").write_text(summary_notes(joined), encoding="utf-8")

    print(f"Wrote {len(joined)} joined rows to {args.out_dir / 'commit_dates.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
