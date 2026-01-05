import pandas as pd
import os

# -----------------------------
# ENGLISH AAC TEST CASES
# -----------------------------
test_data = [
    {"raw_input": "i wnt watr",   "normalized": "I want water",   "category": "Drink",   "pictogram": "water.png",  "generated": "I want water",   "reference": "I need water"},
    {"raw_input": "giv me food",  "normalized": "give me food",   "category": "Food",    "pictogram": "food.png",   "generated": "I want food",   "reference": "I need food"},
    {"raw_input": "turn on ligth","normalized": "turn on light",  "category": "Object",  "pictogram": "light.png",  "generated": "turn on the light", "reference": "switch on light"},
    {"raw_input": "i am hapy",    "normalized": "I am happy",     "category": "Emotion", "pictogram": "happy.png",  "generated": "I am happy",    "reference": "I'm happy"},
    {"raw_input": "it is rainin", "normalized": "It is raining",  "category": "Weather", "pictogram": "rain.png",   "generated": "It is raining", "reference": "It's raining"}
]

# Create dataframe
df = pd.DataFrame(test_data)

# Save to your project Test folder
save_dir = os.path.join(os.getcwd(), "Test")
os.makedirs(save_dir, exist_ok=True)

csv_path = os.path.join(save_dir, "aac_english_test_cases.csv")
df.to_csv(csv_path, index=False, encoding="utf-8")

print("\n✅ AAC English test dataset created successfully!")
print("📁 File saved at:", csv_path)
print("\nSample preview:\n")
print(df)
