import pandas as pd
import numpy as np

# Define columns
columns = [
    "ChallengeID",         # Unique identifier
    "ChallengeName",       # Name of the challenge
    "Category",            # e.g., Computer Vision, NLP, Tabular
    "Participants",        # Number of participants (numerical)
    "PrizeMoney",          # Prize money in USD (numerical)
    "DurationDays",        # Duration in days (numerical)
    "Difficulty",          # Easy, Medium, Hard
    "WinnerTeam",          # Name of winner team
    "Year"                 # Year of the challenge
]

# Generate dummy data
np.random.seed(42)
categories = ["Computer Vision", "NLP", "Tabular", "Time Series", "Recommender"]
difficulties = ["Easy", "Medium", "Hard"]

data = {
    "ChallengeID": np.arange(1, 201),
    "ChallengeName": [f"Kaggle Challenge {i}" for i in range(1, 201)],
    "Category": np.random.choice(categories, 200),
    "Participants": np.random.randint(100, 20000, 200),
    "PrizeMoney": np.random.randint(1000, 100000, 200),
    "DurationDays": np.random.randint(7, 90, 200),
    "Difficulty": np.random.choice(difficulties, 200),
    "WinnerTeam": [f"Team_{np.random.randint(1, 50)}" for _ in range(200)],
    "Year": np.random.choice(range(2015, 2025), 200)
}

df = pd.DataFrame(data)
df.to_excel("./data/kaggle_challenges.xlsx", index=False)
print("Dummy Excel file 'kaggle_challenges.xlsx' created.")

# Read the Excel file
df = pd.read_excel("./data/kaggle_challenges.xlsx")  # Uses pandas.read_excel()[1][2][5]

# (a) List all columns
print("Columns in Excel:")
print(df.columns.tolist())

# (b) Show first 5 rows as sample data
print("\nSample Data (first 5 rows):")
print(df.head())

# (c) Describe numerical columns for quick insights
print("\nNumerical Data Description:")
print(df.describe())

import matplotlib.pyplot as plt

# Histogram of Participants
plt.figure(figsize=(8, 4))
plt.hist(df["Participants"], bins=20, color='skyblue')
plt.title("Distribution of Participants in Kaggle Challenges")
plt.xlabel("Number of Participants")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Boxplot of Prize Money by Difficulty
plt.figure(figsize=(8, 4))
df.boxplot(column="PrizeMoney", by="Difficulty")
plt.title("Prize Money by Difficulty")
plt.suptitle("")
plt.xlabel("Difficulty")
plt.ylabel("Prize Money (USD)")
plt.tight_layout()
plt.show()

# Scatter plot: Participants vs PrizeMoney
plt.figure(figsize=(8, 4))
plt.scatter(df["Participants"], df["PrizeMoney"], alpha=0.6)
plt.title("Participants vs Prize Money")
plt.xlabel("Participants")
plt.ylabel("Prize Money (USD)")
plt.tight_layout()
plt.show()