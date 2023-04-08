import os
import sqlite3
from typing import List, Tuple, TypedDict

SCHEMA_PATH = "music_search/schema.sql"

class AlbumMetadata(TypedDict):
    href: str
    title: str
    artist: str
    genres: str

class AlbumDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        if not os.path.isfile(self.db_path):
            with open (SCHEMA_PATH) as file_handle:
                schema = file_handle.read()

            self.conn = sqlite3.connect(self.db_path)
            self.conn.executescript(schema)
        else:
            self.conn = sqlite3.connect(self.db_path)

        self.conn.row_factory = sqlite3.Row

        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.conn:
            self.conn.commit()
            self.conn.close()

    def insert_album(self, album_metadata: AlbumMetadata, paragraphs: List[str]):
        with self.conn:
            c = self.conn.cursor()

            # Insert metadata
            c.execute("""
                INSERT INTO albums (title, artist, href, genres)
                VALUES (:title, :artist, :href, :genres)
            """, album_metadata)

            album_id = c.lastrowid

            # Insert review paragraphs
            for i, paragraph in enumerate(paragraphs):
                c.execute("""
                    INSERT INTO reviews (album_id, paragraph_num, text)
                    VALUES (:album_id, :paragraph_num, :text)
                """, {'album_id': album_id, 'paragraph_num': i+1, 'text': paragraph})

    def get_albums(self):
        # Query all albums from the database
        with self.conn:
            c = self.conn.cursor()
            c.execute("SELECT * FROM albums")
            rows = c.fetchall()
            
            # Convert the rows to a list of tuples
            albums = [tuple(row) for row in rows]
            
            return albums

    def get_albums_from_paragraph_ids(self, paragraph_ids: List[int]) -> List[Tuple[AlbumMetadata, str]]:
        with self.conn:
            c = self.conn.cursor()
            query = """
            SELECT albums.*, reviews.text
            FROM albums
            INNER JOIN reviews ON reviews.album_id=albums.id
            WHERE reviews.id IN (%s)
            """ % ",".join(
                "?" * len(paragraph_ids)
            )

            c.execute(query, paragraph_ids)
            result = c.fetchall()

        out = [
            (AlbumMetadata(title=r["title"], artist=r["artist"], href=r["href"], genres=r["genres"]), r["text"]) for r in result]

        return out
            
    def get_all_reviews(self):
        with self.conn:
            c = self.conn.cursor()
            sql = """
            SELECT id, text
            FROM reviews
            """
            c.execute(sql)
            rows = c.fetchall()

        return list(zip(*rows))
        
    def get_reviews(self, album_id: int):
        # Query all reviews for the given album ID from the database
        with self.conn:
            c = self.conn.cursor()
            c.execute("SELECT text FROM reviews WHERE album_id = ? ORDER BY paragraph_num", (album_id,))
            rows = c.fetchall()
            
            # Extract the text column from the rows and return as a list
            reviews = [row[0] for row in rows]
            
        return reviews

