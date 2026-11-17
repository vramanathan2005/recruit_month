"""247's class Commits pages: each commit's school (Nov 16 2026: the row's first image is his headshot, titled with his
name, so the school had come out blank for every commit, and committed recruits outside the Composite rankings never
reached Tars: Texas's Raymond Hudson)."""
import unittest

from scrape_247_commit_dates import parse_commits_page

ROW = """<ul><li class="ri-page__list-item"> <div class="wrapper"> <div class="circle-image-block">
<img alt="Raymond Hudson" class="jsonly" data-src="https://s3media.247sports.com/x.jpg" src="https://s3media.247sports.com/Content/Img/1x1.gif" title="Raymond Hudson"/>
<noscript><img alt="Raymond Hudson" src="https://s3media.247sports.com/x.jpg" title="Raymond Hudson"/></noscript> </div>
<div class="recruit"> <a class="ri-page__name-link" href="//247sports.com/Player/raymond-hudson-46164969/" target="_blank">Raymond Hudson</a>
<span class="meta"> Varina (Richmond, VA) </span> <span class="commit-date"> <b>Committed:</b> 10/26/2026 </span> </div>
<div class="metrics"> 6-2 / 215 </div> <div class="rating"> <div class="ri-page__star-and-score"> <span class="score">88</span> </div>
<div class="rank"> <a class="natrank" href="#">NA</a> </div> </div> <div class="position"> Edge </div>
<div class="status"> <img alt="Texas" class="jsonly" src="https://s3media.247sports.com/Content/Img/1x1.gif" title="Texas"/>
<noscript><img alt="Texas" src="https://s3media.247sports.com/t.png" title="Texas"/></noscript> </div> </div> </li></ul>"""


class CommitsPageTest(unittest.TestCase):
    def test_the_school_not_his_headshot(self):
        [row] = parse_commits_page(ROW, 2027)
        self.assertEqual((row.name, row.player_id, row.committed_team, row.committed_date), ("Raymond Hudson", "46164969", "Texas", "10/26/2026"))
        self.assertEqual((row.position, row.rating), ("Edge", "88"))

    def test_no_school_image_is_no_school(self):
        [row] = parse_commits_page(ROW.replace('title="Texas"', 'title="Raymond Hudson"'), 2027)
        self.assertEqual(row.committed_team, "")


if __name__ == "__main__":
    unittest.main()
