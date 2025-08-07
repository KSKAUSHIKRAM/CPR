import os
import sqlite3
from datetime import datetime
class Database:
    def __init__(self, db_name="aac.db"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))  # Current file's dir
        self.db_path = os.path.join(self.base_dir, "..", "Database", db_name)  # ../Database/aac.db
        self.db_path = os.path.abspath(self.db_path)  # Full path
        self.create_table()
    def connect(self):
        return sqlite3.connect("Model/Database/aac.db")
    def create_table(self):
        self.conn=self.connect()
              # e.g., 'Thursday'
        #time = now.strftime('%H:%M:%S') # e.g., "15:22:45"

        cursor=self.conn.execute(
            """
CREATE TABLE IF NOT EXISTS Sentences (
    sentence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    time_phase TEXT,
    location_tag TEXT,
    last_used_at TIME DEFAULT (TIME('now', 'localtime')),
    day TEXT
);

"""

        )
    def insert(self,text,location_tag):
        if isinstance(text,list):
            text=' '.join(text)
        now = datetime.now()
        day = now.strftime('%A') 
        self.cursor=self.conn.cursor()
        self.cursor.execute(
            "INSERT INTO Sentences (text, time_phase, location_tag, day) VALUES (?, ?, ?, ?)",
            (text, "MID-DAY", location_tag, day)
        )        
        self.conn.commit()
    
    def on_close(self):
        if self.conn:
            self.close()
