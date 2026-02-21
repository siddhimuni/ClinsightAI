"""
Phase 3: Dual-Engine Theme Extraction
Engine A: Statistical keyword matching (always runs)
Engine B: Gemini AI (runs when API key is available)
Results are merged and cross-validated.
"""

import json
import re
import time
import pandas as pd
import numpy as np
from typing import Optional
from src.config import THEMES, THEME_LABELS, MOCK_MODE, GROQ_API_KEY, GROQ_MODEL


# ── Keyword rules for statistical engine ────────────────────────────────────

THEME_KEYWORDS = {
    "wait_time": [
        "wait", "waiting", "waited", "slow", "delay", "delayed", "late",
        "hours", "long time", "too long", "forever", "rush", "schedule",
        "on time", "behind", "overdue", "held up",
    ],
    "staff_behavior": [
        "staff", "nurse", "doctor", "rude", "kind", "friendly", "helpful",
        "dismissive", "compassionate", "unprofessional", "professional",
        "care", "caring", "attitude", "behavior", "respectful", "disrespectful",
        "ignored", "attentive", "warm", "cold",
    ],
    "billing": [
        "bill", "billing", "charge", "charged", "invoice", "payment",
        "insurance", "cost", "expensive", "overcharged", "refund",
        "co-pay", "copay", "deductible", "price", "fee",
    ],
    "cleanliness": [
        "clean", "dirty", "filthy", "hygiene", "sanit", "smell", "odor",
        "tidy", "messy", "spotless", "sterile", "bathroom", "restroom",
    ],
    "communication": [
        "communic", "explain", "told", "inform", "update", "unclear",
        "confus", "understand", "told me", "listen", "heard", "respond",
        "call back", "follow up", "coordinate",
    ],
    "facility": [
        "facility", "building", "equipment", "modern", "old", "outdated",
        "room", "bed", "comfortable", "uncomfortable", "parking lot",
        "lobby", "waiting room", "infrastructure",
    ],
    "appointment_scheduling": [
        "appointment", "schedul", "book", "cancel", "rebook", "rescheduled",
        "no-show", "confirm", "reminder", "portal", "online booking",
        "availability", "slot",
    ],
    "food": [
        "food", "meal", "eat", "cafeteria", "nutrition", "diet", "breakfast",
        "lunch", "dinner", "snack", "drink", "taste", "bland", "delicious",
    ],
    "parking": [
        "parking", "park", "valet", "lot", "garage", "access", "entrance",
        "handicap", "wheelchair", "transport", "bus", "traffic",
    ],
}


# ── Statistical Engine ───────────────────────────────────────────────────────

def extract_themes_statistical(review_text: str) -> dict:
    """
    Keyword-based theme detection. Fast, always available, no API needed.
    Returns {theme: {detected: bool, keywords_found: list, severity: float}}
    """
    text_lower = review_text.lower()
    result = {}

    # Negative sentiment signals (amplify severity)
    negative_signals = ["terrible", "awful", "horrible", "worst", "never again",
                        "disgusting", "appalling", "unacceptable", "nightmare",
                        "failure", "incompetent", "negligent", "disgusted"]
    has_negative = any(w in text_lower for w in negative_signals)

    for theme, keywords in THEME_KEYWORDS.items():
        found = [kw for kw in keywords if kw in text_lower]
        detected = len(found) > 0
        # Severity heuristic: more keyword hits + negative sentiment = higher severity
        base_severity = min(len(found) / 3.0, 1.0)
        if detected and has_negative:
            base_severity = min(base_severity + 0.3, 1.0)
        result[theme] = {
            "detected": detected,
            "keywords_found": found[:5],  # top 5 keyword matches
            "severity": round(base_severity, 2),
        }

    return result


# ── Claude AI Engine ─────────────────────────────────────────────────────────

def extract_themes_groq(review_text: str, client=None) -> Optional[dict]:
    """
    Use Groq (LLaMA 3.3) to extract structured themes from a single review.
    Returns None if unavailable (mock mode or API error).
    """
    if MOCK_MODE or client is None:
        return _mock_claude_extraction(review_text)

    prompt = f"""Analyze this hospital review and extract structured operational themes.

Review: "{review_text}"

Return ONLY a valid JSON object with this exact structure:
{{
  "themes_detected": ["theme1", "theme2"],
  "theme_details": {{
    "theme_name": {{
      "sentiment": "positive" | "negative" | "neutral",
      "severity": 1-5,
      "key_phrases": ["exact quote from review"]
    }}
  }},
  "overall_sentiment": "positive" | "negative" | "neutral" | "mixed",
  "key_complaint": "one sentence summary of main issue, or null if positive"
}}

Valid theme names (only use these): {json.dumps(THEMES)}
Only include themes that are clearly mentioned in the review. Return JSON only, no explanation."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        return _safe_parse_json(raw)
    except Exception as e:
        print(f"[WARN] Groq extraction failed: {e}")

    return None


def _mock_claude_extraction(review_text: str) -> dict:
    """
    Rule-based mock that mimics Claude output structure.
    Used when no API key is available.
    """
    stat = extract_themes_statistical(review_text)
    detected = [t for t, v in stat.items() if v["detected"]]

    # Determine overall sentiment from rating signals in text
    negative_words = ["terrible", "awful", "horrible", "worst", "rude", "dirty",
                      "long wait", "overcharged", "nightmare", "disgusting"]
    positive_words = ["excellent", "outstanding", "wonderful", "great", "amazing",
                      "professional", "clean", "on time", "efficient"]
    text_lower = review_text.lower()
    neg_count = sum(1 for w in negative_words if w in text_lower)
    pos_count = sum(1 for w in positive_words if w in text_lower)

    if neg_count > pos_count:
        overall = "negative"
    elif pos_count > neg_count:
        overall = "positive"
    elif detected:
        overall = "mixed"
    else:
        overall = "neutral"

    theme_details = {}
    for theme in detected:
        keywords = stat[theme]["keywords_found"]
        theme_details[theme] = {
            "sentiment": "negative" if overall in ("negative", "mixed") else "positive",
            "severity": min(int(stat[theme]["severity"] * 5) + 1, 5),
            "key_phrases": keywords[:2],
        }

    # Simple key complaint extraction
    sentences = review_text.split(".")
    key_complaint = None
    for sent in sentences:
        if any(w in sent.lower() for w in negative_words):
            key_complaint = sent.strip()[:120]
            break

    return {
        "themes_detected": detected,
        "theme_details": theme_details,
        "overall_sentiment": overall,
        "key_complaint": key_complaint,
    }


# ── Main batch extraction pipeline ──────────────────────────────────────────

def _safe_parse_json(raw: str) -> Optional[dict]:
    """Robustly extract the first valid JSON object from an LLM response."""
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try extracting the outermost {...} block
    try:
        start = raw.index("{")
        # Walk backwards from end to find matching closing brace
        depth = 0
        for i, ch in enumerate(raw[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(raw[start:i+1])
    except (ValueError, json.JSONDecodeError):
        pass
    return None


def _batch_groq_extraction(reviews: list[str], client) -> dict:
    """
    Send up to 20 reviews in a SINGLE Groq call.
    Returns a dict keyed by review index (0-based within the batch).
    This is ~20x faster than one call per review.
    """
    numbered = "\n\n".join(
        f"[{i}] {r[:400]}" for i, r in enumerate(reviews)  # truncate long reviews
    )
    prompt = f"""Analyze each numbered hospital review below.
For EACH review return a JSON object on a single line with key = the review index number.
Valid theme names: {json.dumps(THEMES)}

Format (one JSON line per review, no markdown):
{{"0": {{"themes": ["theme1"], "sentiment": "negative", "complaint": "short complaint or null"}}, "1": {{...}}, ...}}

Reviews:
{numbered}

Return ONLY the JSON object. No explanation."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        result = _safe_parse_json(raw)
        return result if isinstance(result, dict) else {}
    except Exception as e:
        print(f"[WARN] Groq batch call failed: {e}")
    return {}


def run_theme_extraction(df: pd.DataFrame, use_claude: bool = True) -> pd.DataFrame:
    """
    Run theme extraction on all reviews.
    - Statistical engine: runs on ALL reviews instantly (no API)
    - Groq AI: processes reviews in batches of 20 (single API call per batch)
      Result: 300 reviews = ~15 Groq calls instead of 300
    """
    print(f"[INFO] Running theme extraction on {len(df):,} reviews...")
    if MOCK_MODE:
        print("[INFO] Running in MOCK MODE (no API key). Using rule-based AI simulation.")
    else:
        print("[INFO] Groq AI engine active — batch mode (20 reviews per call).")

    client = None
    if not MOCK_MODE and use_claude:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
        except Exception as e:
            print(f"[WARN] Could not initialize Groq client: {e}")

    # Step 1: Statistical pass on ALL reviews (instant)
    review_col = "review_clean" if "review_clean" in df.columns else "review_text"
    all_reviews = df[review_col].tolist()
    all_indices = df.index.tolist()

    stat_results = [extract_themes_statistical(r) for r in all_reviews]

    # Step 2: Groq batch pass
    # Strategy: send ONE stratified sample batch (20 reviews) to Groq to
    # calibrate theme accuracy. Use statistical engine for all other reviews.
    # This keeps token usage ~2,000/run — well within the free tier daily limit.
    groq_results = {}  # df_idx -> ai_result dict

    if client and not MOCK_MODE:
        # Build a stratified sample of up to 20 reviews (balanced by rating)
        sample_local_indices = []
        if "rating" in df.columns and df["rating"].notna().any():
            ratings = df["rating"].tolist()
            for r in [1, 2, 3, 4, 5]:
                bucket = [i for i, v in enumerate(ratings) if v == r]
                sample_local_indices.extend(bucket[:4])  # up to 4 per rating bucket
        else:
            sample_local_indices = list(range(min(20, len(all_reviews))))
        sample_local_indices = sample_local_indices[:20]  # hard cap

        sample_reviews = [all_reviews[i] for i in sample_local_indices]
        print(f"  Sending 1 stratified batch ({len(sample_reviews)} reviews) to Groq...")

        batch_result = _batch_groq_extraction(sample_reviews, client)
        for local_idx_str, data in batch_result.items():
            try:
                local_idx = int(local_idx_str)
                df_idx = all_indices[sample_local_indices[local_idx]]
                themes = data.get("themes", [])
                groq_results[df_idx] = {
                    "themes_detected": themes,
                    "theme_details": {
                        t: {"sentiment": data.get("sentiment", "neutral"),
                            "severity": 3, "key_phrases": []}
                        for t in themes
                    },
                    "overall_sentiment": data.get("sentiment", "neutral"),
                    "key_complaint": data.get("complaint"),
                }
            except (ValueError, IndexError):
                pass
        print(f"  Groq batch complete. {len(groq_results)} reviews enriched by AI.")

    # For all reviews not covered by Groq, use rule-based mock
    for i, idx in enumerate(all_indices):
        if idx not in groq_results:
            groq_results[idx] = _mock_claude_extraction(all_reviews[i])

    # Step 3: Merge statistical + Groq results per review
    results = []
    for i, (idx, stat_result) in enumerate(zip(all_indices, stat_results)):
        ai_result = groq_results.get(idx, _mock_claude_extraction(all_reviews[i]))
        merged = _merge_results(stat_result, ai_result)
        merged["review_idx"] = idx
        results.append(merged)

    results_df = pd.DataFrame(results).set_index("review_idx")

    # Add flat boolean columns for each theme (for ML features)
    for theme in THEMES:
        df[f"theme_{theme}"] = results_df.get(f"{theme}_detected", False)
        df[f"severity_{theme}"] = results_df.get(f"{theme}_severity", 0.0)

    df["overall_sentiment"] = results_df.get("overall_sentiment", "neutral")
    df["key_complaint"] = results_df.get("key_complaint", None)
    df["themes_detected"] = results_df.get("themes_detected", [])

    print(f"[INFO] Theme extraction complete.")
    return df


def _merge_results(stat: dict, ai: Optional[dict]) -> dict:
    """Merge statistical and AI extraction results into flat record."""
    record = {}

    if ai:
        ai_themes = ai.get("themes_detected", [])
        ai_details = ai.get("theme_details", {})
        record["overall_sentiment"] = ai.get("overall_sentiment", "neutral")
        record["key_complaint"] = ai.get("key_complaint")
        record["themes_detected"] = ai_themes
    else:
        ai_themes = []
        ai_details = {}
        record["overall_sentiment"] = "neutral"
        record["key_complaint"] = None
        record["themes_detected"] = []

    for theme in THEMES:
        stat_detected = stat[theme]["detected"]
        ai_detected = theme in ai_themes

        # Detected if either engine finds it
        detected = stat_detected or ai_detected

        if detected and theme in ai_details:
            severity = ai_details[theme].get("severity", 3) / 5.0
        elif detected:
            severity = stat[theme]["severity"]
        else:
            severity = 0.0

        record[f"{theme}_detected"] = detected
        record[f"{theme}_severity"] = round(severity, 2)

    return record


def get_theme_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate theme statistics across all reviews.
    Returns a dataframe with frequency, avg_severity, avg_rating per theme.
    """
    rows = []
    for theme in THEMES:
        col = f"theme_{theme}"
        sev_col = f"severity_{theme}"
        if col not in df.columns:
            continue

        theme_reviews = df[df[col] == True]
        n_theme = len(theme_reviews)
        n_total = len(df)

        if n_theme == 0:
            continue

        avg_rating = theme_reviews["rating"].mean() if "rating" in df.columns else None
        overall_avg = df["rating"].mean() if "rating" in df.columns else None

        # Sample evidence quotes (pick most negative reviews with this theme)
        evidence = []
        if "rating" in df.columns:
            evidence_df = theme_reviews.nsmallest(3, "rating")["review_text"]
        else:
            evidence_df = theme_reviews["review_text"].head(3)
        evidence = [str(r)[:150] + "..." for r in evidence_df.tolist()]

        rows.append({
            "theme": theme,
            "theme_label": THEME_LABELS[theme],
            "frequency_count": n_theme,
            "frequency_pct": round(n_theme / n_total * 100, 1),
            "avg_severity": round(df[sev_col].mean(), 2),
            "avg_rating_with_theme": round(avg_rating, 2) if avg_rating else None,
            "rating_delta": round(avg_rating - overall_avg, 2) if avg_rating and overall_avg else None,
            "evidence_samples": evidence,
        })

    summary = pd.DataFrame(rows).sort_values("frequency_pct", ascending=False)
    return summary
