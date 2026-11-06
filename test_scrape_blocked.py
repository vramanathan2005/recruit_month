#!/usr/bin/env python3
"""247's bot check is never cached or read as an empty page: Nov 5 2026 it was, and the 2027 and 2028 classes vanished
from every output (the flip model then crashed on a class with no rows).

    .venv/bin/python test_scrape_blocked.py
"""
import tempfile
from pathlib import Path

import scrape_247_commit_dates as s

CHALLENGE = "<!DOCTYPE html><html><head><title>Client Challenge</title></head><body></body></html>"


class Response:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class Session:
    def __init__(self, text):
        self.text, self.calls = text, 0
        self.headers = {}

    def get(self, *args, **kwargs):
        self.calls += 1
        return Response(self.text)


def fetcher(cache, text):
    f = s.Fetcher(Path(cache), delay=0)
    f.session = Session(text)
    return f


with tempfile.TemporaryDirectory() as cache:
    f = fetcher(cache, CHALLENGE)
    try:
        f.get(s.BASE + "/Season/2027-Football/Commits/", {"Page": "1"}, refresh=True)
        raise AssertionError("a bot check page was returned as the page")
    except s.Blocked:
        pass
    assert f.session.calls == 1, "a bot check isn't retried"
    assert not list(Path(cache).iterdir()), "a bot check isn't cached"

    # a bot check cached before the guard is no page: it's fetched again
    url, params = s.BASE + "/Season/2027-Football/Commits/", {"Page": "2"}
    s.cache_path_for(Path(cache), url, params).write_text(CHALLENGE)
    f = fetcher(cache, "<html><title>Commits</title></html>")
    assert "Commits" in f.get(url, params, refresh=False) and f.session.calls == 1

    # a good cached page is still used without fetching
    f = fetcher(cache, CHALLENGE)
    assert "Commits" in f.get(url, params, refresh=False) and f.session.calls == 0

print("scrape blocked: ok")
