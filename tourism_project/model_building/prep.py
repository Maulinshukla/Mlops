from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Loads tourism.csv from the repo, cleans it, and writes a stratified
# train/test split that the next pipeline job picks up as an artifact.

DATA_PATH = Path("tourism_project/data/tourism.csv")
TARGET = "ProdTaken"
TEST_SIZE = 0.2
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH}: {df.shape}")

# Identifiers don't carry any behavioural signal, so drop them along with
# the stray index column pandas sometimes leaves behind on export.
drop_cols = [c for c in df.columns if c.startswith("Unnamed")] + ["CustomerID"]
df = df.drop(columns=drop_cols)
print(f"Dropped columns: {drop_cols}")

cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    df[col] = df[col].str.strip()

# 'Fe Male' is a typo for 'Female'. Single and Unmarried are kept apart on
# purpose - their purchase rates differ enough (~36% vs ~24%) that merging
# them would throw away real signal.
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {n_before - len(df)} duplicate rows")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Stratified so the ~19% positive rate holds in both sets.
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Impute AFTER the split, using train statistics only, then apply those same
# values to test. Doing this on the full dataset before splitting would leak
# test-set information into training. This dataset has no missing values
# today so the loops below are a no-op, but it keeps the split leakage-free
# if a future data refresh isn't as clean.
numeric_features = [c for c in X.columns if c not in cat_cols]
for col in numeric_features:
    if Xtrain[col].isna().any():
        fill_value = Xtrain[col].median()
        Xtrain[col] = Xtrain[col].fillna(fill_value)
        Xtest[col] = Xtest[col].fillna(fill_value)
for col in cat_cols:
    if Xtrain[col].isna().any():
        fill_value = Xtrain[col].mode()[0]
        Xtrain[col] = Xtrain[col].fillna(fill_value)
        Xtest[col] = Xtest[col].fillna(fill_value)
print(f"Missing values after cleaning: {int(Xtrain.isna().sum().sum() + Xtest.isna().sum().sum())}")

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Train: {Xtrain.shape}, positive rate {ytrain.mean():.3f}")
print(f"Test : {Xtest.shape}, positive rate {ytest.mean():.3f}")
print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
