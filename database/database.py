import sqlite3
from datetime import datetime


DB_NAME = "crosswalk.db"


def create_database():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        object_type TEXT,
        confidence REAL,
        status TEXT,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()



def save_detection(object_type, confidence, status):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO detections
    (object_type, confidence, status, timestamp)

    VALUES (?, ?, ?, ?)
    """,
    (
        object_type,
        confidence,
        status,
        datetime.now()
    ))

    conn.commit()
    conn.close()