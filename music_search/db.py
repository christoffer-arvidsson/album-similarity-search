import os
import sqlite3
from typing import Dict, List, Tuple, TypedDict

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


# def create_connection(path):
#     """Create connection to database."""
#     connection = None
#     try:
#         connection = sqlite3.connect(path)
#         logging.info("Connected to SQLite DB successfully")
#     except Error as e:
#         logging.error(f"Unable to connect to DB with error: {e}")

#     return connection


# def execute_query(connection, query):
#     """Execute SQL query."""
#     cursor = connection.cursor()
#     try:
#         cursor.execute(query)
#         connection.commit()
#         logging.info("Query successfully")
#     except Error as e:
#         logging.error(f"Query failed with error: {e}")


# def execute_get_query(connection, query):
#     """Execute SQL query."""
#     cursor = connection.cursor()
#     try:
#         cursor.execute(query)
#         result = cursor.fetchall()
#         logging.info("Query successfully")
#     except Error as e:
#         logging.error(f"Query failed with error: {e}")

#     return result


# def execute_many_query(connection, query, records):
#     """Execute query across many records."""
#     cursor = connection.cursor()
#     try:
#         cursor.executemany(query, records)
#         connection.commit()
#         logging.info("Query successfully")
#     except Error as e:
#         logging.error(f"Query failed with error: {e}")


# def create_tables(connection):
#     sent_query = """
#     CREATE TABLE IF NOT EXISTS sentences (
#       sent_id INTEGER PRIMARY KEY,
#       doc_id INTEGER NOT NULL,
#       sentence TEXT
#     );
#     """

#     doc_query = """
#     CREATE TABLE IF NOT EXISTS documents (
#       doc_id INTEGER PRIMARY KEY,
#       link TEXT
#     );
#     """

#     execute_query(connection, sent_query)
#     execute_query(connection, doc_query)


# def drop_table(connection, table_name):
#     query = f"DROP TABLE {table_name}"
#     execute_query(connection, query)


# def get_documents_from_sentence_ids(connection, sent_ids):
#     cursor = connection.cursor()

#     query = """
#     SELECT sentences.*, documents.link
#     FROM sentences
#     INNER JOIN documents ON sentences.doc_id=documents.doc_id
#     WHERE sentences.sent_id IN (%s)
#     """ % ",".join(
#         "?" * len(sent_ids)
#     )

#     result = None
#     try:
#         cursor.execute(query, sent_ids)
#         result = cursor.fetchall()
#     except Error as e:
#         logging.error(f"Could not select {sent_ids} due to error: {e}")

#     return result


# def insert_sentences(connection, sent_ids, doc_ids, sentences):
#     add_sent_query = "INSERT INTO sentences VALUES(?,?,?);"
#     records = list(zip(sent_ids, doc_ids, sentences))

#     execute_many_query(connection, add_sent_query, records)


# def insert_documents(connection, doc_ids, doc_paths):
#     add_docs_query = "INSERT INTO documents VALUES(?,?);"
#     records = list(zip(doc_ids, doc_paths))

#     execute_many_query(connection, add_docs_query, records)
