import sqlite3
from datetime import datetime
import os

# Create database folder if not exists
os.makedirs("../database", exist_ok=True)

DB_PATH = "../database/uc1_events.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    event_type TEXT
)
""")

conn.commit()

def log_event(event_type):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        "INSERT INTO events (timestamp, event_type) VALUES (?, ?)",
        (timestamp, event_type)
    )

    conn.commit()

    print(f"[LOG] {timestamp} -> {event_type}")