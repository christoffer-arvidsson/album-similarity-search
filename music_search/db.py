import logging
import sqlite3
from sqlite3 import Error


def create_connection(path):
    """Create connection to database."""
    connection = None
    try:
        connection = sqlite3.connect(path)
        logging.info("Connected to SQLite DB successfully")
    except Error as e:
        logging.error(f"Unable to connect to DB with error: {e}")

    return connection

def execute_query(connection, query):
    """Execute SQL query."""
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        logging.info("Query successfully")
    except Error as e:
        logging.error(f"Query failed with error: {e}")

def execute_get_query(connection, query):
    """Execute SQL query."""
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        logging.info("Query successfully")
    except Error as e:
        logging.error(f"Query failed with error: {e}")

    return result


def execute_many_query(connection, query, records):
    """Execute query across many records."""
    cursor = connection.cursor()
    try:
        cursor.executemany(query, records)
        connection.commit()
        logging.info("Query successfully")
    except Error as e:
        logging.error(f"Query failed with error: {e}")

def create_tables(connection):
    sent_query = """
    CREATE TABLE IF NOT EXISTS sentences (
      sent_id INTEGER PRIMARY KEY,
      doc_id INTEGER NOT NULL,
      sentence TEXT
    );
    """

    doc_query = """
    CREATE TABLE IF NOT EXISTS documents (
      doc_id INTEGER PRIMARY KEY,
      link TEXT
    );
    """

    execute_query(connection, sent_query)
    execute_query(connection, doc_query)

def drop_table(connection, table_name):
    query = f"DROP TABLE {table_name}"
    execute_query(connection, query)

def get_documents_from_sentence_ids(connection, sent_ids):
    cursor = connection.cursor()

    query = """
    SELECT sentences.*, documents.link
    FROM sentences
    FULL JOIN documents ON sentences.doc_id=documents.doc_id
    WHERE sentences.sent_id IN (%s)
    """ % ",".join("?" * len(sent_ids))

    result = None
    try:
        cursor.execute(query, sent_ids)
        result = cursor.fetchall()
    except Error as e:
        logging.error(f"Could not select {sent_ids} due to error: {e}")


    return result

def insert_sentences(connection, sent_ids, doc_ids, sentences):
    add_sent_query = "INSERT INTO sentences VALUES(?,?,?);"
    records = list(zip(sent_ids, doc_ids, sentences))

    execute_many_query(connection, add_sent_query, records)

def insert_documents(connection, doc_ids, doc_paths):
    add_docs_query = "INSERT INTO documents VALUES(?,?);"
    records = list(zip(doc_ids, doc_paths))

    execute_many_query(connection, add_docs_query, records)
