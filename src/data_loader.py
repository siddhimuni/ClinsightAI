"""
Phase 1 & 2: Data Ingestion, EDA, and Preprocessing
Handles loading the Kaggle hospital reviews dataset and cleaning it.
"""

import pandas as pd
import numpy as np
import re
import os


# ── Column name candidates (Kaggle dataset may use different names) ──────────
REVIEW_COL_CANDIDATES = ["Feedback", "feedback", "review_text", "review", "text", "comments", "description", "Review", "Text"]
RATING_COL_CANDIDATES = ["Ratings", "ratings", "star_rating", "rating", "stars", "score", "Rating", "Stars"]
HOSPITAL_COL_CANDIDATES = ["hospital_name", "hospital", "name", "facility", "Hospital", "Name"]
DATE_COL_CANDIDATES = ["date", "review_date", "created_at", "Date", "Review_Date"]


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the hospital reviews CSV and normalize column names."""
    if not os.path.exists(filepath):
        print(f"[INFO] Dataset not found at '{filepath}'. Loading sample data instead.")
        return _generate_sample_data()

    df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
    print(f"[INFO] Loaded {len(df):,} reviews from {filepath}")

    df = _normalize_columns(df)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map whatever column names the CSV has to our standard names."""
    col_map = {}

    for candidate in REVIEW_COL_CANDIDATES:
        if candidate in df.columns:
            col_map[candidate] = "review_text"
            break

    for candidate in RATING_COL_CANDIDATES:
        if candidate in df.columns:
            col_map[candidate] = "rating"
            break

    for candidate in HOSPITAL_COL_CANDIDATES:
        if candidate in df.columns:
            col_map[candidate] = "hospital_name"
            break

    for candidate in DATE_COL_CANDIDATES:
        if candidate in df.columns:
            col_map[candidate] = "date"
            break

    df = df.rename(columns=col_map)

    if "review_text" not in df.columns:
        raise ValueError(f"Could not find a review text column. Columns found: {list(df.columns)}")
    if "rating" not in df.columns:
        print("[WARN] No rating column found — rating analysis will be limited.")
        df["rating"] = np.nan

    if "hospital_name" not in df.columns:
        df["hospital_name"] = "Unknown Hospital"

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the dataframe for analysis."""
    original_len = len(df)

    # Drop rows with missing review text
    df = df.dropna(subset=["review_text"])
    df["review_text"] = df["review_text"].astype(str)

    # Drop rows where review is too short to be meaningful
    df = df[df["review_text"].str.len() >= 20]

    # Clean review text
    df["review_clean"] = df["review_text"].apply(_clean_text)

    # Normalize ratings to 1–5 scale
    if df["rating"].notna().any():
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        # Handle 0–10 scale
        if df["rating"].max() > 5:
            df["rating"] = (df["rating"] / 2).round(1)
        df["rating"] = df["rating"].clip(1, 5)

    # Add derived features
    df["review_length"] = df["review_text"].str.len()
    df["word_count"] = df["review_text"].str.split().str.len()

    # Parse dates if available
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year_month"] = df["date"].dt.to_period("M")

    # Reset index
    df = df.reset_index(drop=True)

    print(f"[INFO] Preprocessing: {original_len:,} -> {len(df):,} reviews kept")
    return df


def _clean_text(text: str) -> str:
    """Remove noise while preserving sentiment-bearing words."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Lowercase
    text = text.lower().strip()
    return text


def get_eda_stats(df: pd.DataFrame) -> dict:
    """Return key EDA statistics for the dashboard."""
    stats = {
        "total_reviews": len(df),
        "avg_rating": round(df["rating"].mean(), 2) if df["rating"].notna().any() else None,
        "rating_distribution": df["rating"].value_counts().sort_index().to_dict() if df["rating"].notna().any() else {},
        "avg_review_length": int(df["review_length"].mean()),
        "unique_hospitals": df["hospital_name"].nunique() if "hospital_name" in df.columns else 1,
    }

    if "date" in df.columns and df["date"].notna().any():
        stats["date_range"] = {
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        }

    # Rating buckets
    if df["rating"].notna().any():
        stats["pct_negative"] = round(len(df[df["rating"] <= 2]) / len(df) * 100, 1)
        stats["pct_positive"] = round(len(df[df["rating"] >= 4]) / len(df) * 100, 1)
        stats["pct_neutral"] = round(100 - stats["pct_negative"] - stats["pct_positive"], 1)

    return stats


# ── Sample data for demo / testing without the real dataset ─────────────────

def _generate_sample_data(n: int = 300) -> pd.DataFrame:
    """Generate realistic sample hospital review data for demo purposes."""
    np.random.seed(42)

    sample_reviews = [
        # 1-star reviews
        ("I waited over 3 hours past my appointment time. The staff at the front desk were rude and dismissive when I asked for updates. The waiting room was dirty and smelled bad.", 1),
        ("Billing department is a nightmare. They charged me twice for the same procedure and it took 6 months to get a refund. No one returns calls.", 1),
        ("The nurse was incredibly rude and condescending. I felt ignored and humiliated. I will never return to this hospital.", 1),
        ("Appointment scheduled for 9am, saw the doctor at 12:30pm with no explanation. The facility looks run down and unclean.", 1),
        ("They lost my paperwork twice. Communication between departments is terrible. Left more confused than when I arrived.", 1),
        ("Waited 2 hours in the emergency room without anyone checking on me. The place was filthy and the staff seemed overwhelmed.", 1),
        ("Insurance billing was completely wrong. Still disputing charges 8 months later. No one in billing can explain the charges.", 1),
        ("The doctor spent less than 5 minutes with me after a 90-minute wait. Felt like a cattle drive, not healthcare.", 1),

        # 2-star reviews
        ("Wait times are consistently too long. I understand they are busy but 90 minutes past scheduled time is unacceptable.", 2),
        ("Parking is terrible and very expensive. The facility itself is okay but getting there is a nightmare.", 2),
        ("Some staff are kind but others are dismissive. Inconsistent experience depending on who you get.", 2),
        ("The food in the cafeteria was awful and overpriced. The medical care was adequate but nothing special.", 2),
        ("Scheduling is a mess. My appointment was moved three times without proper notification.", 2),

        # 3-star reviews
        ("Average experience. Wait time was about an hour which I expected. Staff were professional if not warm.", 3),
        ("The facility is clean and modern. Wait times could be better. The doctors are knowledgeable.", 3),
        ("Mixed experience. The nursing staff were excellent but the administrative side was confusing.", 3),
        ("Decent care overall. Parking is a hassle and the wait was long but the doctor was thorough.", 3),

        # 4-star reviews
        ("Very professional staff. The nurse explained everything clearly. Only giving 4 stars because the wait was 45 minutes.", 4),
        ("Great doctors who really listened to my concerns. Clean facility, easy parking. Would recommend.", 4),
        ("Excellent communication from the entire team. My care coordinator kept me informed at every step.", 4),
        ("The staff were friendly and caring. The facility was spotless. Scheduling was easy online.", 4),

        # 5-star reviews
        ("Outstanding care from start to finish. The team was compassionate, professional, and efficient. Zero wait time.", 5),
        ("Best medical experience I have ever had. The doctor took time to explain everything. Highly recommend.", 5),
        ("Exceptional staff, clean facility, on-time appointments. My care was personalized and thorough.", 5),
        ("The nurses and doctors here genuinely care about patients. Communication was excellent. I felt safe and heard.", 5),
        ("Incredibly efficient and caring. My appointment started on time, the doctor was brilliant, and billing was clear.", 5),
    ]

    rows = []
    hospitals = ["St. Mary Medical Center", "General Hospital", "City Health Clinic", "Riverside Medical"]

    for _ in range(n):
        review, base_rating = sample_reviews[np.random.randint(len(sample_reviews))]
        # Add some noise to ratings
        noisy_rating = np.clip(base_rating + np.random.randint(-1, 2), 1, 5)
        rows.append({
            "review_text": review,
            "rating": float(noisy_rating),
            "hospital_name": hospitals[np.random.randint(len(hospitals))],
            "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(np.random.randint(0, 730))),
        })

    df = pd.DataFrame(rows)
    print(f"[INFO] Generated {len(df)} sample reviews for demo. Add real data at data/hospital_reviews.csv")
    return df
