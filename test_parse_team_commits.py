#!/usr/bin/env python3
"""A team's own commits page lists every commit, rated or not; the class-wide page leaves unrated ones off, so Raymond
Hudson (Texas, from Clemson, Oct 26 2026) was on no page Tars read (Nov 17 2026).

    .venv/bin/python test_parse_team_commits.py
"""
import scrape_247_commit_dates as s

ITEM = ('<li class="ri-page__list-item"><div class="wrapper"><div class="recruit"><a class="ri-page__name-link" '
        'href="//247sports.com/Player/raymond-hudson-46164969/">Raymond Hudson</a><span class="meta"> Varina (Richmond, VA) </span></div>'
        '<div class="metrics"> 6-1 / 190 </div><div class="rating"><span class="score">88</span><a class="natrank">NA</a></div>'
        '<div class="status"><p class="commit-date withDate"> Commit 10/26/2026 </p></div><div class="position"> S </div></div></li>')
rows = s.parse_team_commits_page(f'<ul class="ri-page__list">{ITEM}</ul>', 2027, "Texas")
assert [(r.player_id, r.name, r.committed_team, r.committed_date, r.high_school, r.position) for r in rows] == [
    ("46164969", "Raymond Hudson", "Texas", "10/26/2026", "Varina (Richmond, VA)", "S")], rows
assert s.parse_commits_page(f'<ul>{ITEM}</ul>', 2027) == [], "the class-page parser never read team pages"

INDEX = ('<a href="https://247sports.com/college/texas/season/2027-football/commits/">Texas</a>'
         '<a href="https://247sports.com/college/texas/season/2027-football/commits/">22 Commits</a>'
         '<a href="https://247sports.com/college/texas-am/season/2027-football/commits/">304.81</a>'
         '<a href="https://247sports.com/college/texas-am/season/2027-football/commits/">Texas A&amp;M</a>')
assert s.parse_team_index(INDEX, 2027) == [("texas", "Texas"), ("texas-am", "Texas A&M")], s.parse_team_index(INDEX, 2027)
print("parse team commits: ok")
