#!/usr/bin/env python3
"""Build a local player-ID to 247Sports headshot URL index from cached pages (the daily refresh runs it after the 247
scrape, so newly ranked recruits get their photos). Players whose cached page is gone keep the URL they had."""

from __future__ import annotations

import csv
import html
import re
from pathlib import Path

from bs4 import BeautifulSoup


CACHE_DIR = Path("data/cache")
OUTPUT_PATH = Path("data/player_images.csv")
PLAYER_ID_RE = re.compile(r"/player/[^/]+-(\d+)", re.IGNORECASE)


def image_url_for_item(item) -> str:
    image = item.select_one("div.circle-image-block img")
    if not image:
        return ""
    return html.unescape(image.get("data-src") or image.get("src") or "")


def main() -> int:
    images: dict[str, str] = {}
    found: dict[str, str] = {}
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(encoding="utf-8") as handle:
            images = {row["player_id"]: row["image_url"] for row in csv.DictReader(handle) if row.get("image_url")}
    for path in CACHE_DIR.glob("*.html"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "circle-image-block" not in text or "/player/" not in text.lower():
            continue
        soup = BeautifulSoup(text, "html.parser")
        for item in soup.select("li.rankings-page__list-item, li.ri-page__list-item, li.commitment-list__item"):
            link = item.find("a", href=PLAYER_ID_RE)
            if not link:
                continue
            match = PLAYER_ID_RE.search(link.get("href", ""))
            image_url = image_url_for_item(item)
            if match and image_url and "1x1.gif" not in image_url:
                found.setdefault(match.group(1), image_url)
    images.update(found)   # a fresh URL from today's pages replaces the old one

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["player_id", "image_url"])
        writer.writeheader()
        writer.writerows(
            {"player_id": player_id, "image_url": image_url}
            for player_id, image_url in sorted(images.items())
        )
    print(f"Wrote {len(images)} player image URLs to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
