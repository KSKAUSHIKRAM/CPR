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

    def TFIDF_recommend(self, sentences, top_k=5):
        # ✅ Deduplicate first (preserve order)
        unique_sentences = list(dict.fromkeys(sentences))

        if not unique_sentences:
            return []

        # TF-IDF on unique sentences only
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(unique_sentences)
        tfidf_sums = tfidf_matrix.sum(axis=1).A1

        # Frequency counts on unique sentences
        sentence_counts = Counter(unique_sentences)
        freq_values = np.array([sentence_counts[s] for s in unique_sentences], dtype=float)

        # Normalize
        freq_norm = (freq_values - freq_values.min()) / (freq_values.max() - freq_values.min() + 1e-9)
        tfidf_norm = (tfidf_sums - tfidf_sums.min()) / (tfidf_sums.max() - tfidf_sums.min() + 1e-9)

        # Combine scores
        alpha = 0.6
        combined_scores = alpha * freq_norm + (1 - alpha) * tfidf_norm

        # Sort indices by score (descending)
        ranked_indices = np.argsort(combined_scores)[::-1]

        # Pick top_k
        top_indices = ranked_indices[:top_k]
        top_sentences = [unique_sentences[i] for i in top_indices]

        # Debug print
        for idx in top_indices:
            print(f"{combined_scores[idx]:.4f}  {unique_sentences[idx]} "
                f"(freq: {freq_values[idx]}, tfidf_sum: {tfidf_sums[idx]:.4f})")

        return top_sentences


    def last_sentence(self,location_tag):
        current_phase=self.get_time_phase()

        #last_id = self.cursor.lastrowid
        #print(f"Last inserted sentence ID: {last_id}")
        self.cursor.execute("SELECT text from Sentences where location_tag = ? AND time_phase = ? ORDER BY sentence_id DESC LIMIT 1", (location_tag, current_phase))
        last_row = self.cursor.fetchone()
        return last_row
    def recommend(self, location_tag):
        sentences = []
        current_phase=self.get_time_phase()
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "SELECT  text FROM Sentences WHERE location_tag = ? AND time_phase = ?",
            (location_tag, current_phase)
        )
        rows_1= self.cursor.fetchall()
        sentences_loc_match= [row[0] for row in rows_1]
        sentences_match1=self.TFIDF_recommend(sentences_loc_match)
        self.cursor.execute(
            "SELECT text FROM Sentences WHERE time_phase = ? and location_tag <> ?",
            (current_phase,location_tag)
        )
        rows_2= self.cursor.fetchall()
        sentences_unmatch_loc= [row[0] for row in rows_2]
        sentences_match2=self.TFIDF_recommend(sentences_unmatch_loc)
        sentences=sentences_match1 + sentences_match2
        if not sentences:
            print("No sentences found.")
            return
        else:
            return sentences

    
    def on_close(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
