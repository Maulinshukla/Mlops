"""
Data Registration
-----------------
Validates the tourism dataset that lives inside the GitHub repository
(tourism_project/data/tourism.csv) and prints a short summary.

The script exits with a non-zero status if validation fails, which makes the
GitHub Actions job fail fast before any downstream step runs.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

DATA_PATH = Path("tourism_project/data/tourism.csv")
TARGET = "ProdTaken"

# Columns defined in the project's data dictionary
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def file_md5(path: Path) -> str:
    """Fingerprint of the dataset version being registered."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    # 1. The file must exist in the repository
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # 2. All expected columns must be present
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        sys.exit(f"ERROR: missing expected columns: {missing_cols}")
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    # 3. The target must be binary (0/1)
    bad_labels = set(df[TARGET].dropna().unique()) - {0, 1}
    if bad_labels:
        sys.exit(f"ERROR: unexpected target values in {TARGET}: {bad_labels}")

    # 4. Dataset must not be empty
    if df.empty:
        sys.exit("ERROR: dataset is empty")

    # ---------------- Summary ----------------
    print("=" * 60)
    print("DATASET REGISTRATION SUMMARY")
    print("=" * 60)
    print(f"Source file        : {DATA_PATH}")
    print(f"Version (MD5)      : {file_md5(DATA_PATH)}")
    print(f"Rows x Columns     : {df.shape[0]} x {df.shape[1]}")
    print(f"Expected columns   : all {len(EXPECTED_COLUMNS)} present")
    print(f"Extra columns      : {extra_cols if extra_cols else 'none'}")
    print(f"Missing values     : {int(df.isna().sum().sum())}")
    print(f"Duplicate CustomerIDs: {int(df['CustomerID'].duplicated().sum())}")

    print("\nTarget distribution (ProdTaken):")
    print(df[TARGET].value_counts().rename("count").to_frame()
          .assign(share=lambda t: (t["count"] / t["count"].sum()).round(3)))

    print("\nCategorical levels:")
    for col in df.select_dtypes(include="object").columns:
        print(f"  {col:<15}: {sorted(df[col].unique())}")

    print("\nNumeric summary:")
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c not in ("CustomerID", TARGET) and not c.startswith("Unnamed")]
    print(df[num_cols].describe().T[["mean", "min", "max"]].round(2))

    print("\nDataset validated and registered successfully.")


if __name__ == "__main__":
    main()
