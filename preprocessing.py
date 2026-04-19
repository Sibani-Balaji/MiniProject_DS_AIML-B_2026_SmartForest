"""
SmartForest - Forest Fire Risk Prediction
preprocessing.py - Data Cleaning and Preprocessing Module
SRM Institute of Science and Technology | AIML-B | 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

# ─────────────────────────────────────────
# STEP 1: Load Raw Dataset
# ─────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the Algerian Forest Fires dataset."""
    df = pd.read_csv(filepath, skiprows=1)
    df.columns = df.columns.str.strip()
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────
# STEP 2: Clean Data
# ─────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove nulls, fix types, drop region header rows."""
    # Drop rows that are region header labels
    df = df[~df['day'].str.contains('day|Region', na=True, case=False)]
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Strip whitespace from all string columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Convert numeric columns
    numeric_cols = ['day', 'month', 'year', 'Temperature', 'RH', 'Ws',
                    'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()
    print(f"[INFO] After cleaning: {df.shape[0]} rows remaining")
    return df


# ─────────────────────────────────────────
# STEP 3: Encode Target Variable
# ─────────────────────────────────────────

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode 'Classes' column: fire=1, not fire=0."""
    df['Classes'] = df['Classes'].str.strip().str.lower()
    df['Classes'] = df['Classes'].map({'fire': 1, 'not fire': 0})
    print(f"[INFO] Class distribution:\n{df['Classes'].value_counts()}")
    return df


# ─────────────────────────────────────────
# STEP 4: Feature Selection
# ─────────────────────────────────────────

def select_features(df: pd.DataFrame):
    """Select features and target for model training."""
    feature_cols = ['Temperature', 'RH', 'Ws', 'Rain',
                    'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
    X = df[feature_cols]
    y = df['Classes']
    return X, y


# ─────────────────────────────────────────
# STEP 5: Scale + Split
# ─────────────────────────────────────────

def scale_and_split(X, y, test_size=0.2, random_state=42):
    """Standardize features and split into train/test sets."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler


# ─────────────────────────────────────────
# STEP 6: Save Processed Data
# ─────────────────────────────────────────

def save_processed(df: pd.DataFrame, output_path: str):
    """Save cleaned dataframe to processed_data folder."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Processed data saved to: {output_path}")


# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────

if __name__ == "__main__":
    raw_path = "../dataset/raw_data/Algerian_forest_fires_dataset.csv"
    processed_path = "../dataset/processed_data/cleaned_forest_fires.csv"

    df = load_data(raw_path)
    df = clean_data(df)
    df = encode_target(df)
    save_processed(df, processed_path)

    X, y = select_features(df)
    X_train, X_test, y_train, y_test, scaler = scale_and_split(X, y)
    print("[INFO] Preprocessing complete. Ready for model training.")
