import os
import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

class Database:
    def __init__(self, db_name="aac.db"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))  # Current file's directory
        self.db_path = os.path.join(self.base_dir, "Database", db_name)  # Model/Database/aac.db
        self.db_path = os.path.abspath(self.db_path)  # Convert to full path
        print("Database path:", self.db_path)  # Debugging
        self.create_table()
    def connect(self):
        return sqlite3.connect(self.db_path)

    def create_table(self):
        self.conn = self.connect()
        self.conn.execute(
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
        self.conn.commit()

    def insert(self, text, location_tag):
        if isinstance(text, list):
            text = ' '.join(text)
        now = datetime.now()
        day = now.strftime('%A') 
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "INSERT INTO Sentences (text, location_tag, day) VALUES (?, ?, ?)",
            (text, location_tag, day)
        )        
        self.conn.commit()
    def get_time_phase(self):
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        h = hour + minute/60  # fractional hour

        if 6 <= h < 9.9833:   # 09:59 is 9 + 59/60 ≈ 9.9833
            return "Morning"
        elif 10 <= h < 13.9833:  # 13:59 ≈ 13.9833
            return "Midday"
        elif 14 <= h < 17.9833:  # 17:59 ≈ 17.9833
            return "Afternoon"
        elif 18 <= h < 20.9833:  # 20:59 ≈ 20.9833
            return "Evening"
        else:
            return "Night"


    def recommend(self, location_tag):
        current_phase=self.get_time_phase()
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "SELECT text FROM Sentences WHERE location_tag = ? AND time_phase = ?",
            (location_tag, current_phase)
        )
        rows = self.cursor.fetchall()
        sentences = [row[0] for row in rows]
        if not sentences:
            print("No sentences found.")
            return

    # Calculate TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)

    # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()

    # Display TF-IDF for each sentence
        for i, sentence in enumerate(sentences):
            print(f"\nSentence: {sentence}")
            for col in tfidf_matrix[i].nonzero()[1]:
                print(f"  {feature_names[col]}: {tfidf_matrix[i, col]:.4f}")

            for row in rows:
                print(row)

    def on_close(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
