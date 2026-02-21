# HackVerse 2026 | ClinsightAI

**Company Track:** Clinsight
**Team Name:** Neural Ninjas
**Team Members:** Abhiram M V, Siddhi Muni, Jayan Agarwal

---

## 1. Problem Statement

**Which company problem did you choose?**

ClinsightAI — AI-Powered Healthcare Review Intelligence System.

Multi-location healthcare groups collect thousands of patient reviews, but this data sits unused because it is unstructured, noisy, and hard to translate into operational action. Hospital administrators cannot easily answer: *Which specific operational problems are hurting our ratings? Are they systemic or isolated? What should we fix first?*

**End user:** Hospital operations managers and healthcare business owners who need to make data-driven decisions about where to invest in improvement.

**Why it matters:** Patient reviews contain high-signal intelligence about wait times, staff behavior, billing issues, and facility quality — but only if analyzed systematically. A 0.5-star rating improvement can directly impact patient retention and hospital reputation.

**Key assumptions:**
- Reviews are honest reflections of patient experience
- Star ratings correlate with operational quality
- Themes extracted from text are actionable by hospital management
- The Kaggle dataset is representative of real hospital review patterns

---

## 2. Why We Chose This Problem

- **Business impact is tangible:** Unlike abstract ML benchmarks, this problem has a direct line from analysis to action — hospital managers can use the output tomorrow
- **Technical depth is layered:** The problem requires NLP (theme extraction), statistical modeling (impact quantification), classification (systemic detection), and LLM integration (roadmap generation) — not just a single model call
- **Hybrid approach opportunity:** We could combine semantic embeddings with regression — exactly the kind of hybrid approach the rubric encourages
- **Explainability matters:** Healthcare decisions require transparent, evidence-backed reasoning — not black-box predictions

---

## 3. Solution Overview

ClinsightAI is a 7-step pipeline that transforms raw hospital reviews into a structured intelligence report:

1. **Input:** Hospital review CSV (text + star ratings)
2. **Processing:** Embedding-based theme matching, Ridge regression impact scoring, systemic classification
3. **Output:** Structured JSON with ranked themes, severity scores, confidence intervals, 1-star vs 5-star segment analysis, and an LLM-generated improvement roadmap
4. **What makes it unique:** We use sentence embeddings (not keyword lists) for theme detection, bootstrap confidence intervals for statistical rigor, and a clear separation between data-driven analysis and LLM-generated recommendations

---

## 4. Architecture & System Design

```
Hospital Reviews CSV
        |
   [1] Data Loader ──────── Clean text, normalize ratings, extract features
        |
   [2] Theme Extractor ──── Sentence embeddings (all-MiniLM-L6-v2) + cosine similarity
        |
   [3] Impact Quantifier ── Ridge regression + bootstrap confidence intervals
        |
   [4] Systemic Detector ── Frequency + impact thresholds → SYSTEMIC / MODERATE / ISOLATED
        |
   [5] Roadmap Generator ── Groq LLM (LLaMA 3.1) → prioritized recommendations
        |
   [6] Evaluation ───────── 8 test cases, precision/recall/F1, visualization plots
        |
   [7] Structured Output ── JSON report + Streamlit dashboard
```

**Why this architecture?**

Each module is independent, testable, and has a clear single responsibility. The pipeline flows from raw data to business-ready output without circular dependencies. The separation means we can swap any component (e.g., replace Ridge with XGBoost, or swap the embedding model) without affecting the rest.

**Trade-offs considered:**
- **Embeddings vs keyword matching:** Embeddings are more robust to paraphrasing but require a model download (~80MB). We chose embeddings because keyword lists are not defensible — the model generalizes to unseen phrasings.
- **Ridge vs Random Forest:** Ridge gives interpretable coefficients ("+0.2 stars from this theme") while Random Forest gives feature importance but not direction. We chose Ridge for explainability.
- **LLM for roadmap only:** We use the LLM (Groq) only for generating natural language recommendations — not for theme detection or scoring. This keeps the analytical core deterministic and reproducible.

---

## 5. Data Handling & Preprocessing

**Dataset:** [Kaggle Hospital Reviews Dataset](https://www.kaggle.com/datasets/junaid6731/hospital-reviews-dataset) — 1,001 reviews with feedback text, sentiment labels, and star ratings (1-5).

**Preprocessing steps:**
1. Rename CSV columns to snake_case (`Feedback` → `review_text`, `Ratings` → `rating`)
2. Drop reviews shorter than 20 characters (noise)
3. Clean text: strip HTML tags, URLs, extra whitespace, lowercase
4. Normalize ratings to 1-5 scale
5. Extract features: `review_length`, `word_count`

**After preprocessing:** 991 reviews retained (99.5% retention rate — minimal data loss).

**Limitations:**
- No date column — trend analysis over time is not possible
- No hospital name column — cross-hospital comparison not available
- Single-source dataset — may not generalize to other regions or hospital types

---

## 6. Modeling & AI Strategy

### Theme Extraction — Sentence Embeddings + Cosine Similarity

**Model:** `all-MiniLM-L6-v2` (22M parameters, ~80MB, runs on CPU)

**How it works:**
1. Define each theme as a short phrase (e.g., `"long waiting time and delays"`)
2. Encode theme descriptions and all reviews into the same 384-dimensional embedding space
3. Compute cosine similarity between each review and each theme
4. If similarity > 0.3, assign the theme

**Why this model?**
- Lightweight enough to run on any laptop (no GPU required)
- Pre-trained on 1B+ sentence pairs — strong semantic understanding
- Deterministic — same input always produces same output (unlike LLM-based extraction)

**Alternatives considered:**
- **Keyword matching:** Fragile, requires maintaining large keyword lists, misses paraphrases
- **BERTopic / LDA:** Unsupervised topic discovery — themes are unnamed and may not align with operational categories
- **Zero-shot classification (Hugging Face):** Slower, requires larger models, and our approach achieves similar results with a simpler architecture
- **LLM-based extraction:** Non-deterministic, expensive, and overkill for a classification task

### Impact Quantification — Ridge Regression

**Model:** `Ridge(alpha=1.0)` — L2-regularized linear regression

**How it works:**
- Features: 9 binary theme columns (detected / not detected)
- Target: star rating
- Output: per-theme coefficient (e.g., `-0.208` for Wait Time = associated with 0.2-star rating drop)

**Confidence:** Bootstrap resampling (100 iterations). Confidence = proportion of bootstrap samples where the coefficient sign matches the main estimate. High confidence (>0.9) means the theme consistently helps or hurts ratings.

**Why Ridge?**
- Coefficients are directly interpretable as "how much this theme shifts the rating"
- L2 regularization handles correlated themes without overfitting
- Cross-validated R² measures overall model fit

**Alternatives considered:**
- **Random Forest:** Good for importance ranking but coefficients are not interpretable
- **SHAP:** Expensive to compute, and our Ridge coefficients already give directional impact
- **Correlation:** Too simplistic — doesn't control for confounding themes

### Roadmap Generation — Groq LLM

**Model:** `llama-3.1-8b-instant` via Groq API

**Prompt strategy:** We provide the LLM with only the structured impact data (top 5 themes, their frequency, rating impact, severity) and ask for 5-7 prioritized recommendations. The LLM never sees raw reviews — it operates on our validated analytics output.

**Grounding method:** The LLM recommendations are grounded by the quantitative analysis. Each recommendation references a specific theme with measured impact. The system works without the LLM (returns empty roadmap) — the core analysis is fully deterministic.

**Limitations:**
- LLM recommendations are suggestions, not validated actions
- Quality depends on the Groq model's understanding of healthcare operations
- Runs in demo mode (empty roadmap) without an API key

---

## 7. Evaluation & Metrics

### Theme Extraction Accuracy

**8 hand-crafted test cases** covering all 9 themes with known expected outputs.

| Metric | Score |
|--------|-------|
| **Avg Precision** | 0.791 |
| **Avg Recall** | 0.666 |
| **Avg F1** | 0.709 |
| **Perfect Match Rate** | 50% |

**What these metrics measure:**
- **Precision:** Of the themes we detected, how many were correct? (0.79 = 79% of detections are correct)
- **Recall:** Of the themes that should have been detected, how many did we find? (0.67 = we catch 2/3 of expected themes)
- **F1:** Harmonic mean of precision and recall — our overall accuracy

**Limitations:** Test cases are hand-labeled and may reflect our own biases. A larger, independently labeled test set would be more rigorous.

### Evaluation Visualizations

The pipeline generates two visualization files in `outputs/`:

1. **`similarity_distribution.png`** — Cosine similarity distribution for each theme across 200 sampled reviews. Shows how well the 0.3 threshold separates matching vs non-matching reviews.
2. **`regression_diagnostics.png`** — Ridge regression coefficients with bootstrap confidence bars per theme. Shows which themes reliably help or hurt ratings.

### Regression Model Quality

- **Cross-validated R²:** Measures how much of the rating variance is explained by theme presence alone
- **Per-theme bootstrap confidence:** 100 resampling iterations measuring coefficient sign stability (>0.9 = highly confident)

---

## 8. Business Impact & Actionability

### What the output tells a hospital manager:

1. **"What are patients complaining about?"** → Theme frequency ranking (Communication: 29%, Staff Behavior: 28.6%, Wait Time: 7.9%)
2. **"Which complaints actually hurt our ratings?"** → Rating impact scores (Wait Time: -0.208 stars, Billing: -0.192 stars)
3. **"Are these one-off complaints or systemic?"** → Issue classification (SYSTEMIC / MODERATE / ISOLATED)
4. **"What should we fix first?"** → Severity score (frequency x impact, normalized 0-1) + prioritized roadmap
5. **"What do 1-star reviews look like vs 5-star?"** → Rating segment analysis showing theme prevalence by rating tier

### Real-world usability:

A hospital COO could open the JSON output or dashboard and immediately see:
- Wait Time appears in 15.7% of 1-star reviews vs 3.5% of 5-star reviews → clear target
- Billing has the strongest negative impact (-0.192 stars) with 100% bootstrap confidence → high priority
- Communication is frequent (29%) but has positive impact (+0.146) → patients praise it, don't fix what isn't broken

### Limitations:
- Analysis is correlational, not causal — a theme being associated with low ratings doesn't mean fixing it will improve ratings
- Recommendations are AI-generated suggestions requiring domain expert validation
- Dataset is a single snapshot — continuous monitoring would require regular re-runs

---

## 9. Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **NLP** | sentence-transformers (all-MiniLM-L6-v2) |
| **ML** | scikit-learn (Ridge regression, StandardScaler) |
| **LLM** | Groq API (LLaMA 3.1 8B) |
| **Dashboard** | Streamlit + Plotly |
| **Data** | pandas, numpy |
| **Visualization** | matplotlib, plotly |
| **Config** | python-dotenv |

---

## 10. How to Run the Project

### Clone Repository
```bash
git clone <repo-link>
cd ClinsightAI
```

### Install Dependencies
```bash
conda create -n clinsight-nlp python=3.10
conda activate clinsight-nlp
pip install -r requirements.txt
```

### Run the Pipeline
```bash
python main.py
```
This generates `outputs/results.json` and evaluation visualizations.

### Run the Dashboard
```bash
streamlit run dashboard/app.py
```

### (Optional) Enable LLM Roadmap
Create a `.env` file with:
```
GROQ_API_KEY=your_key_here
```

---

## 11. Repository Structure

```
ClinsightAI/
├── data/
│   └── hospital.csv              # Kaggle hospital reviews dataset
├── src/
│   ├── config.py                 # Themes, thresholds, API config
│   ├── data_loader.py            # CSV loading and preprocessing
│   ├── theme_extractor.py        # Embedding-based theme matching
│   ├── impact_quantifier.py      # Ridge regression + bootstrap confidence
│   ├── systemic_detector.py      # SYSTEMIC / MODERATE / ISOLATED classification
│   ├── roadmap_generator.py      # Groq LLM roadmap + executive summary
│   └── evaluation.py             # Test cases, metrics, visualization
├── dashboard/
│   └── app.py                    # Streamlit dashboard (7 pages)
├── outputs/
│   ├── results.json              # Structured JSON output
│   ├── similarity_distribution.png
│   └── regression_diagnostics.png
├── main.py                       # Main pipeline entry point
├── requirements.txt
└── README.md
```

---

## 12. Alignment with HackVerse Rubric

| Rubric Category | How We Address It |
|---|---|
| **Problem Understanding (10pts)** | Clear business framing — who the user is, why it matters, measurable goals, assumptions stated |
| **Data & System Design (15pts)** | Meaningful EDA (distributions, anomalies), clean architecture with 7-stage pipeline, architecture diagram above |
| **Technical Depth (20pts)** | Goes beyond sentiment — embedding-based theme matching, regression impact scoring, systemic classification, 1-star vs 5-star segmentation |
| **Modeling Strategy (20pts)** | Hybrid approach: semantic embeddings + Ridge regression + LLM. Each model choice explained with alternatives considered |
| **Evaluation & Metrics (15pts)** | 8 test cases, precision/recall/F1 metrics, 2 evaluation visualizations, metric explanations and limitations |
| **Business Actionability (10pts)** | Structured JSON matching problem statement format, prioritized recommendations, decision-ready insights |
| **Visualization & UX (5pts)** | 7-page Streamlit dashboard with EDA, theme analysis, impact matrix, systemic detection, roadmap, raw data explorer |
| **Demo & Communication (5pts)** | Live dashboard demo, clear pipeline execution |

---

## Compliance Statement

We confirm that this project was developed during HackVerse 2026.
We used only permitted datasets and tools.
No private code sharing occurred between teams.
All work is original.
