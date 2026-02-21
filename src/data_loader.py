"""
Data loading and preprocessing for hospital reviews.
"""

import pandas as pd
import numpy as np
import re


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load hospital reviews CSV and rename columns to snake_case."""
    df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
    print(f"[INFO] Loaded {len(df):,} reviews from {filepath}")

    df = df.rename(columns={
        "Feedback": "review_text",
        "Sentiment Label": "sentiment_label",
        "Ratings": "rating",
    })

    # Drop the empty unnamed column if present
    df = df.loc[:, df.columns.str.strip() != ""]

    if "review_text" not in df.columns:
        raise ValueError(f"No 'Feedback' column found. Columns: {list(df.columns)}")

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text, normalize ratings, and add derived features."""
    original_len = len(df)

    df = df.dropna(subset=["review_text"])
    df["review_text"] = df["review_text"].astype(str)
    df = df[df["review_text"].str.len() >= 20]

    df["review_clean"] = df["review_text"].apply(clean_text)

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating"] = df["rating"].clip(1, 5)

    df["review_length"] = df["review_text"].str.len()
    df["word_count"] = df["review_text"].str.split().str.len()

    df = df.reset_index(drop=True)
    print(f"[INFO] Preprocessing: {original_len:,} -> {len(df):,} reviews kept")
    return df


def get_eda_stats(df: pd.DataFrame) -> dict:
    """Return summary statistics for the dashboard."""
    has_ratings = df["rating"].notna().any()
    total = len(df)

    stats = {
        "total_reviews": total,
        "avg_rating": round(df["rating"].mean(), 2) if has_ratings else None,
        "rating_distribution": df["rating"].value_counts().sort_index().to_dict() if has_ratings else {},
        "avg_review_length": int(df["review_length"].mean()),
    }

    if has_ratings:
        stats["pct_negative"] = round(len(df[df["rating"] <= 2]) / total * 100, 1)
        stats["pct_positive"] = round(len(df[df["rating"] >= 4]) / total * 100, 1)
        stats["pct_neutral"] = round(100 - stats["pct_negative"] - stats["pct_positive"], 1)

    return stats


def clean_text(text: str) -> str:
    """Strip HTML, URLs, extra whitespace, and lowercase."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


if __name__ == "__main__":
    import json

    df = load_dataset("data/hospital.csv")
    df = preprocess(df)
    stats = get_eda_stats(df)

    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
    print(f"\nEDA Stats:\n{json.dumps(stats, indent=2)}")
