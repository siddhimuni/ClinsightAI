"""
Theme extraction using sentence embeddings.

Each theme is defined by a single short phrase. Reviews are matched to themes
via cosine similarity between their embeddings — no hardcoded keyword lists.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import THEMES, THEME_LABELS

# Each theme is defined by one short phrase — the embedding model handles matching
THEME_DESCRIPTIONS = {
    "wait_time": "long waiting time and delays",
    "staff_behavior": "staff and nurse behavior and attitude",
    "billing": "billing charges and insurance cost",
    "cleanliness": "cleanliness hygiene and sanitation",
    "communication": "communication and sharing information with patient",
    "facility": "facility building and equipment condition",
    "appointment_scheduling": "appointment scheduling and booking",
    "food": "food and meal quality",
    "parking": "parking and transportation access",
}

SIMILARITY_THRESHOLD = 0.3


def _load_model():
    """Load a lightweight sentence embedding model (~80MB, runs on CPU)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def run_theme_extraction(df: pd.DataFrame, use_claude: bool = True) -> pd.DataFrame:
    """
    Embed all reviews and theme descriptions, then assign themes
    based on cosine similarity. No keyword lists, no API needed.
    """
    print(f"[INFO] Running embedding-based theme extraction on {len(df):,} reviews...")

    model = _load_model()

    theme_names = list(THEME_DESCRIPTIONS.keys())
    theme_texts = list(THEME_DESCRIPTIONS.values())
    theme_embeddings = model.encode(theme_texts)

    review_col = "review_clean" if "review_clean" in df.columns else "review_text"
    review_embeddings = model.encode(df[review_col].tolist(), show_progress_bar=True)

    # similarity matrix: (n_reviews, n_themes)
    sim_matrix = cosine_similarity(review_embeddings, theme_embeddings)

    for i, theme in enumerate(theme_names):
        scores = sim_matrix[:, i]
        df[f"theme_{theme}"] = scores >= SIMILARITY_THRESHOLD
        df[f"severity_{theme}"] = np.where(scores >= SIMILARITY_THRESHOLD, np.round(scores, 2), 0.0)

    df["themes_detected"] = [
        [theme_names[j] for j in range(len(theme_names)) if sim_matrix[i, j] >= SIMILARITY_THRESHOLD]
        for i in range(len(df))
    ]

    df["overall_sentiment"] = df["rating"].apply(_rating_to_sentiment)
    df["key_complaint"] = None

    print(f"[INFO] Theme extraction complete.")
    return df


def _rating_to_sentiment(rating):
    if pd.isna(rating):
        return "neutral"
    if rating <= 2:
        return "negative"
    if rating >= 4:
        return "positive"
    return "neutral"


def get_theme_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate theme frequency, severity, and rating delta across all reviews."""
    rows = []
    overall_avg = df["rating"].mean()

    for theme in THEMES:
        col = f"theme_{theme}"
        sev_col = f"severity_{theme}"
        if col not in df.columns:
            continue

        theme_reviews = df[df[col] == True]
        if len(theme_reviews) == 0:
            continue

        avg_rating = theme_reviews["rating"].mean()

        evidence = theme_reviews.nsmallest(3, "rating")["review_text"]
        evidence = [str(r)[:150] + "..." for r in evidence.tolist()]

        rows.append({
            "theme": theme,
            "theme_label": THEME_LABELS[theme],
            "frequency_count": len(theme_reviews),
            "frequency_pct": round(len(theme_reviews) / len(df) * 100, 1),
            "avg_severity": round(df[sev_col].mean(), 2),
            "avg_rating_with_theme": round(avg_rating, 2),
            "rating_delta": round(avg_rating - overall_avg, 2),
            "evidence_samples": evidence,
        })

    return pd.DataFrame(rows).sort_values("frequency_pct", ascending=False)


if __name__ == "__main__":
    from src.data_loader import load_dataset, preprocess

    df = load_dataset("data/hospital.csv")
    df = preprocess(df)
    df = run_theme_extraction(df)
    summary = get_theme_summary(df)

    print("\nTheme Summary:")
    print(summary[["theme_label", "frequency_pct", "avg_severity", "rating_delta"]].to_string(index=False))
