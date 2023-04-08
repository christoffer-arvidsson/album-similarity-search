
from time import sleep
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import trange

from music_search.db import AlbumDatabase, AlbumMetadata
from music_search.util import get_config

DB_PATH = "data/albums.db"
BASE_URL = "https://www.pitchfork.com"

# 12 albums per page
N_PAGES = 100
N_ALBUMS = N_PAGES * 12

def scrape_review(album_href: str) -> Optional[List[str]]:
    res = requests.get(album_href)
    soup = BeautifulSoup(res.text, "html.parser")
    review_blocks = soup.find_all("div", class_="body__inner-container")

    if review_blocks is not None:
        return [r.text for b in review_blocks for r in b.find_all("p")[:-1]]
    else:
        return None

def scrape_best_albums(db_path: str):
    prog_bar = trange(N_PAGES, desc="Scraping", leave=True)
    for p in prog_bar:
        link = f"https://pitchfork.com/best/high-scoring-albums/?page={p+1}"
        prog_bar.set_description(f"Scraping {link}")
        try: 
            res = requests.get(link)
            res.raise_for_status()

            soup = BeautifulSoup(res.text, "html.parser")

            hits = soup.find_all("div", class_="review")
            for album in hits:
                href = f"{BASE_URL}{album.find('a', class_='review__link')['href']}"
                title = album.find("h2", class_="review__title-album").get_text()
                artist = album.find("ul", class_="artist-list review__title-artist").get_text()

                genres = album.find("ul", class_="review__genre-list")
                genres = genres.get_text() if genres else ""

                album_metadata = AlbumMetadata(href=href, title=title, artist=artist, genres=genres)
                # Todo: skip scraping if it already exists in database
                paragraphs = scrape_review(href)
                if paragraphs is not None:
                    with AlbumDatabase(db_path) as db_handler:
                        db_handler.insert_album(album_metadata, paragraphs)
                else:
                    raise Exception("Could not parse review body.")

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except Exception as err:
            print(f"An error occurred: {err}")

        sleep(0.5)
    

def main():
    config = get_config("./configs/default.yaml")
    scrape_best_albums(config.db_path)



if __name__ == '__main__':
    main()
