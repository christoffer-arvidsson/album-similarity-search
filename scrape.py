
import json
import os
from time import sleep

import requests
from bs4 import BeautifulSoup
from tqdm import trange, tqdm

BASE_URL = "https://www.pitchfork.com"

# 12 albums per page
N_PAGES = 10
N_ALBUMS = N_PAGES * 12

def parse_album(album_href: str):
    res = requests.get(album_href)
    soup = BeautifulSoup(res.text, "html.parser")
    review_blocks = soup.find_all("div", class_="body__inner-container")

    review = ""
    if review_blocks is not None:
        for b in review_blocks:
            review += "\n".join([r.text for r in b.find_all("p")[:-1]])

    else:
        return None

    return {
        "title": soup.find("h1").text,
        "href": album_href,
        "review": review,
    }


def get_album_links():
    album_links = []
    prog_bar = trange(N_PAGES, desc="Scraping", leave=True)
    for p in prog_bar:
        link = f"https://pitchfork.com/best/high-scoring-albums/?page={p+1}"
        prog_bar.set_description(f"Scraping {link}")
        res = requests.get(link)
        soup = BeautifulSoup(res.text, "html.parser")

        hits = soup.find_all("div", class_="review")
        for album in hits:
            href = f"{BASE_URL}{album.find('a', class_='review__link')['href']}"
            album_links.append(href)

        sleep(0.5)

    return album_links
    

def main():
    album_links = get_album_links()

    albums = []
    for l in tqdm(album_links):
        albums.append(parse_album(l))
        
    try:
        os.makedirs("data")
    except FileExistsError:
        # directory already exists
        pass

    with open("data/album_metadata.json", "w") as file_handle:
        json.dump(albums, file_handle, indent=2)



if __name__ == '__main__':
    main()
