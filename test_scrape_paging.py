#!/usr/bin/env python3
"""247 orders players tied on a rating differently from one request to the next, so its ranking pages overlap: a player
shows on two pages and another on none. Nov 17 2026: 111 repeats on 2027's Composite rankings and ~100 ranked commits
never read (no timeline, no flip odds). scrape_paged downloads again until it has read as many players as rows.

    .venv/bin/python test_scrape_paging.py
"""
from types import SimpleNamespace

import scrape_247_commit_dates as s

PLAYERS = [str(i) for i in range(10)]       # 10 players, pages of 4: 4 + 4 + 2


class ShufflingFetcher:
    """Page 2's first request repeats page 1's last player in place of player 4 (the overlap 247 serves); later requests
    serve the true order. Records each request's refresh so the test can see the extra passes skip the cache."""

    def __init__(self, overlaps: int):
        self.overlaps, self.refreshes = overlaps, []

    def refresh_for(self, year):
        return True

    def get(self, url, params=None, refresh=None):
        self.refreshes.append(refresh)
        page = int(params["Page"])
        ids = PLAYERS[(page - 1) * 4:page * 4]
        if page == 2 and self.overlaps:
            self.overlaps -= 1
            ids = ["3"] + ids[1:]
        return ",".join(ids)


def parser(html, year):
    return [SimpleNamespace(player_id=i, class_year=year) for i in html.split(",") if i]


fetcher = ShufflingFetcher(overlaps=1)
rows = s.scrape_paged(fetcher, "u", 2027, parser, "rankings", None)
assert sorted(r.player_id for r in rows) == PLAYERS, [r.player_id for r in rows]
assert s.FORCE in fetcher.refreshes, "the second pass must skip the cache, or it rereads the overlapping pages"

# 247 never settles: the scrape stops after REPAGE_PASSES with what it has (and says so), no repeats
fetcher = ShufflingFetcher(overlaps=99)
rows = s.scrape_paged(fetcher, "u", 2027, parser, "rankings", None)
assert [r.player_id for r in rows] == [p for p in PLAYERS if p != "4"], [r.player_id for r in rows]
assert fetcher.refreshes.count(s.FORCE) == (s.REPAGE_PASSES - 1) * 4

# no overlap: one pass, nothing more downloaded
fetcher = ShufflingFetcher(overlaps=0)
assert len(s.scrape_paged(fetcher, "u", 2027, parser, "rankings", None)) == 10 and s.FORCE not in fetcher.refreshes
print("scrape paging: ok")
