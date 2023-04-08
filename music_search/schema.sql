
-- Create a table to store the album metadata
CREATE TABLE albums (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    href TEXT NOT NULL,
    genres TEXT NOT NULL
);

-- Create a table to store the album reviews
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    album_id INTEGER NOT NULL,
    paragraph_num INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (album_id) REFERENCES albums(id)
);
