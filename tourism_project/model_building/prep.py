"""
Data Preparation
----------------
1. Loads the dataset directly from the repository data folder.
2. Cleans the data and removes columns with no predictive value.
3. Splits into stratified train / test sets and saves them locally as CSVs
   (Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv) in the working directory.
   GitHub Actions uploads these files as a workflow artifact for the training job.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("tourism_project/data/tourism.csv")
TARGET = "ProdTaken"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------- Load ----------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH}: {df.shape}")

# ---------------- Clean ----------------
# a) Drop identifiers / index artefacts: they carry no information about
#    purchase behaviour and would only add noise (or leak row order).
drop_cols = [c for c in df.columns if c.startswith("Unnamed")] + ["CustomerID"]
df = df.drop(columns=drop_cols)
print(f"Dropped columns: {drop_cols}")

# b) Normalise text values (strip stray whitespace)
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    df[col] = df[col].str.strip()

# c) Fix inconsistent labels: 'Fe Male' is a typo of 'Female'.
#    NOTE: 'Single' and 'Unmarried' are kept separate on purpose - they have
#    very different purchase rates in the data, so merging would lose signal.
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# d) Remove exact duplicate customer records (after dropping IDs) so the same
#    record cannot appear in both train and test sets.
n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {n_before - len(df)} duplicate rows")

# e) Impute any missing values (defensive - makes the pipeline robust to
#    future data refreshes): median for numeric, mode for categorical.
num_cols = [c for c in df.columns if c not in cat_cols + [TARGET]]
for col in num_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode()[0])
print(f"Missing values after cleaning: {int(df.isna().sum().sum())}")

# ---------------- Split ----------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Stratify to keep the ~19% positive rate identical in both splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ---------------- Save locally ----------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Train: {Xtrain.shape}, positive rate {ytrain.mean():.3f}")
print(f"Test : {Xtest.shape}, positive rate {ytest.mean():.3f}")
print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
