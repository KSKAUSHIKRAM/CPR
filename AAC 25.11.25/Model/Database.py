import os
import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np


class Database:
    def __init__(self, db_name="aac.db"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))  # Current file's directory
        self.db_path = os.path.join(self.base_dir, "Database", db_name)  # Model/Database/aac.db
        self.db_path = os.path.abspath(self.db_path)  # Convert to full path
        print("Database path:", self.db_path)  # Debugging

        self.create_table()

    def create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
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
            conn.commit()

    def insert(self, text, location_tag):
        if isinstance(text, list):
            text = ' '.join(text)
        now = datetime.now()
        day = now.strftime('%A')
        time_phase = self.get_time_phase()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO Sentences (text, location_tag, day, time_phase) VALUES (?, ?, ?, ?)",
                (text, location_tag, day, time_phase)
            )
            conn.commit()

    def get_time_phase(self):
        now = datetime.now()
        h = now.hour + now.minute / 60  # fractional hour

        if 6 <= h < 9.9833:       # 09:59
            return "Morning"
        elif 10 <= h < 13.9833:   # 13:59
            return "Midday"
        elif 14 <= h < 17.9833:   # 17:59
            return "Afternoon"
        elif 18 <= h < 20.9833:   # 20:59
            return "Evening"
        else:
            return "Night"

    def TFIDF_recommend(self, sentences, top_k=5):
        unique_sentences = list(dict.fromkeys(sentences))
        if not unique_sentences:
            return []

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(unique_sentences)
        tfidf_sums = tfidf_matrix.sum(axis=1).A1

        sentence_counts = Counter(unique_sentences)
        freq_values = np.array([sentence_counts[s] for s in unique_sentences], dtype=float)

        freq_norm = (freq_values - freq_values.min()) / (freq_values.max() - freq_values.min() + 1e-9)
        tfidf_norm = (tfidf_sums - tfidf_sums.min()) / (tfidf_sums.max() - tfidf_sums.min() + 1e-9)

        alpha = 0.6
        combined_scores = alpha * freq_norm + (1 - alpha) * tfidf_norm

        ranked_indices = np.argsort(combined_scores)[::-1]
        top_indices = ranked_indices[:top_k]
        top_sentences = [unique_sentences[i] for i in top_indices]

        for idx in top_indices:
            print(f"{combined_scores[idx]:.4f}  {unique_sentences[idx]} "
                  f"(freq: {freq_values[idx]}, tfidf_sum: {tfidf_sums[idx]:.4f})")

        return top_sentences

    def last_sentence(self, location_tag):
        current_phase = self.get_time_phase()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT text FROM Sentences WHERE location_tag = ? AND time_phase = ? "
                "ORDER BY sentence_id DESC LIMIT 1",
                (location_tag, current_phase)
            )
            last_row = cursor.fetchone()
            return last_row[0] if last_row else None

    def location_matched(self, location_tag):
        current_phase = self.get_time_phase()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT text FROM Sentences WHERE location_tag = ? AND time_phase = ?",
                (location_tag, current_phase)
            )
            rows = cursor.fetchall()
        sentences = [row[0] for row in rows]
        print("location_matched sentences:", sentences)  # Debugging
        return self.TFIDF_recommend(sentences)

    def location_unmatched(self, location_tag):
        current_phase = self.get_time_phase()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT text FROM Sentences WHERE time_phase = ? AND location_tag <> ?",
                (current_phase, location_tag)
            )
            rows = cursor.fetchall()
        sentences = [row[0] for row in rows]
        print("location_unmatched sentences:", sentences)
        return self.TFIDF_recommend(sentences)
    def time_phase_unmatched(self, location_tag):
        current_phase = self.get_time_phase()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT text FROM Sentences WHERE location_tag = ? and time_phase <> ?",
                (location_tag,current_phase)
            )
            rows = cursor.fetchall()
        sentences = [row[0] for row in rows]
        print("time_phase_unmatched sentences:", sentences)
        return self.TFIDF_recommend(sentences)

    def on_close(self):
        # ✅ No persistent connection, nothing to close
        pass
