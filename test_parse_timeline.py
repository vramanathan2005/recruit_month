#!/usr/bin/env python3
"""A timeline page saved by a browser (Tars's browser collector) reads like one fetched directly: Chrome splits 247's
<p> into an empty one and the event's text, and on Nov 6 2026 every 2027 and 2028 timeline read as empty (the flip
model then crashed on a class with no rows).

    .venv/bin/python test_parse_timeline.py
"""
import scrape_247_commit_dates as s

RECRUIT = s.RankingRow(2027, "46145095", "Jalen Brewster", "https://247sports.com/player/jalen-brewster-46145095/", *[""] * 12)


def page(item: str) -> str:
    return f'<html><body><section class="main"><ul class="timeline-event-index_lst"><li>{item}</li></ul></section></body></html>'


direct = page('<div class="left"><b>June 11, 2026: Official Visit</b><p>Jalen Brewster officially visits Florida Gators</p></div>')
browser = page('<div class="left"> <b>June 11, 2026: Official Visit</b> <p> </p><p>Jalen Brewster officially visits Florida Gators</p> <p></p> </div>')
for html in (direct, browser):
    rows = s.parse_timeline_events(html, RECRUIT)
    assert [(r.event_type, r.event_date, r.school) for r in rows] == [("Official Visit", "6/11/2026", "Florida Gators")], rows

print("parse timeline: ok")
