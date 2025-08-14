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


    def recommend(self, location_tag):
        current_phase=self.get_time_phase()
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "SELECT text FROM Sentences WHERE location_tag = ? AND time_phase = ?",
            (location_tag, current_phase)
        )
        rows = self.cursor.fetchall()
        sentences = [row[0] for row in rows]
        sentence_counts = Counter(sentences)

# Extract unique sentences to match TF-IDF order
        unique_sentences = list(sentence_counts.keys())
        if not sentences:
            print("No sentences found.")
            return

    # Calculate TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(unique_sentences)
        tfidf_sums = tfidf_matrix.sum(axis=1).A1
    # Get feature names (words)
    #feature_names = vectorizer.get_feature_names_out()
        freq_values = np.array([sentence_counts[s] for s in unique_sentences], dtype=float)
        freq_norm = (freq_values - freq_values.min()) / (freq_values.max() - freq_values.min() + 1e-9)
        tfidf_norm = (tfidf_sums - tfidf_sums.min()) / (tfidf_sums.max() - tfidf_sums.min() + 1e-9)

# 5. Combine scores with alpha weight (0 ≤ alpha ≤ 1)
        alpha = 0.6
        combined_scores = alpha * freq_norm + (1 - alpha) * tfidf_norm

# 6. Rank sentences by combined score descending
        ranked_indices = np.argsort(combined_scores)[::-1]

    # Display TF-IDF for each sentence
        """ for i, sentence in enumerate(sentences):
                print(f"\nSentence: {sentence}")
                for col in tfidf_matrix[i].nonzero()[1]:
                    print(f"  {feature_names[col]}: {tfidf_matrix[i, col]:.4f}")

            for row in rows:
                print(row)"""
        for idx in ranked_indices:
            print(f"{combined_scores[idx]:.4f}  {unique_sentences[idx]} (freq: {freq_values[idx]}, tfidf_sum: {tfidf_sums[idx]:.4f})")


    def on_close(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()