# HackVerse 2026 | ClinsightAI

**Company Track:** Clinsight
**Team Name:** Neural Ninjas
**Team Members:** Abhiram M V, Siddhi Muni, Jayan Agarwal

---

## 1. Problem Statement

**Which company problem did you choose?**

ClinsightAI — AI-Powered Healthcare Review Intelligence System.

**Restated in our own words:**

Multi-location healthcare groups collect thousands of patient reviews across platforms, but this data sits unused because it is unstructured, noisy, and impossible to translate into operational action at scale. Hospital administrators today cannot answer three critical questions: *Which specific operational problems are hurting our ratings? Are those problems systemic or isolated incidents? What should we fix first to maximize rating improvement?*

**End user:** Hospital operations managers, COOs, and healthcare business owners who need to make data-driven decisions about where to invest in quality improvement.

**Why this problem is important:**

Patient reviews contain high-signal intelligence about wait times, staff behavior, billing disputes, facility quality, and communication breakdowns — but only if analyzed systematically. Industry data shows a 0.5-star rating improvement can directly impact patient acquisition, retention, and hospital reputation. Yet most healthcare organizations lack the tooling to extract structured, prioritized insights from free-text feedback.

**Key assumptions:**
- Reviews are honest reflections of patient experience
- Star ratings correlate with operational quality
- Themes extracted from text are actionable by hospital management
- The Kaggle dataset is representative of real hospital review patterns
- Operational categories (Wait Time, Billing, Staff Behavior, etc.) are meaningful to hospital decision-makers

**What success looks like:**
- A hospital manager can open the output and immediately identify the top 3 operational areas hurting their ratings
- Every insight is backed by quantitative evidence (coefficients, confidence intervals, frequency data)
- Recommendations are prioritized by measurable severity, not intuition

---

## 2. Why We Chose This Problem

- **Business impact is tangible:** Unlike abstract ML benchmarks, this problem has a direct line from analysis to action — hospital managers can use the output tomorrow to make budget and staffing decisions
- **Technical depth is layered:** The problem requires NLP (theme extraction), statistical modeling (impact quantification), classification (systemic detection), and LLM integration (roadmap generation) — not just a single model call
- **Hybrid approach opportunity:** We could combine semantic embeddings with Ridge regression and variance-based analysis — exactly the kind of multi-signal hybrid approach the rubric encourages
- **Explainability is non-negotiable:** Healthcare decisions require transparent, evidence-backed reasoning — not black-box predictions. This constraint pushed us toward interpretable models (Ridge over Random Forest, embeddings over LLMs for classification)
- **Real problem, real data:** The Kaggle hospital reviews dataset contains authentic patient feedback, making our analysis directly transferable to production hospital analytics systems

---

## 3. Solution Overview

ClinsightAI is a **7-step modular pipeline** that transforms raw hospital reviews into a structured intelligence report with prioritized improvement recommendations.

- **Input:** Hospital review CSV containing free-text feedback and star ratings (1-5)
- **Processing:** Embedding-based semantic theme matching, Ridge regression for impact quantification with bootstrap confidence intervals, variance-based systemic classification, and LLM-grounded roadmap generation
- **Output:** Structured JSON report with ranked themes, severity scores, confidence intervals, 1-star vs 5-star segment analysis, and an AI-generated improvement roadmap — plus a 7-page interactive Streamlit dashboard
- **What makes it unique:** We use sentence embeddings (not keyword lists) for theme detection, bootstrap confidence intervals for statistical rigor, coefficient-of-variation-based consistency analysis for systemic detection, and a clear separation between deterministic data-driven analysis and LLM-generated recommendations. The system works fully without the LLM — the analytical core is 100% reproducible.

---

## 4. Architecture & System Design

### Architecture Diagram

![ClinsightAI Architecture](docs/architecture_diagram.png)

### Pipeline Description

```
Hospital Reviews CSV (1,001 reviews)
        │
   [1] Data Loader ──────────── Clean text, normalize ratings, extract features
        │                        Output: 991 reviews (99.5% retention)
        │
   [2] Theme Extractor ──────── Sentence embeddings (all-MiniLM-L6-v2) + cosine similarity
        │                        9 operational themes matched semantically
        │
   [3] Impact Quantifier ────── Ridge regression (α=1.0) + bootstrap CI (100 iterations)
        │                        Per-theme rating impact coefficients
        │
   [4] Systemic Detector ────── Consistency (40%) + Frequency (30%) + Impact (30%)
        │                        Classification: SYSTEMIC / MODERATE / ISOLATED
        │
   [5] Roadmap Generator ────── Groq LLM (LLaMA 3.1 8B) → prioritized recommendations
        │                        LLM sees only structured data, never raw reviews
        │
   [6] Evaluation Engine ────── 8 test cases, precision/recall/F1, 2 visualization plots
        │
        ├── [7a] Structured JSON Report (results.json)
        └── [7b] Streamlit Dashboard (7 interactive pages)
```

### Data Flow

1. **Raw CSV** → `data_loader.py` cleans text (strip HTML/URLs, lowercase), normalizes ratings to 1-5, drops noise (<20 chars), extracts derived features (review_length, word_count)
2. **Cleaned DataFrame** → `theme_extractor.py` encodes all reviews + 9 theme descriptions into 384-dimensional embedding space, computes cosine similarity matrix (991 × 9), assigns themes at threshold > 0.3
3. **Theme-tagged DataFrame** → `impact_quantifier.py` fits Ridge regression (features = 9 binary theme columns, target = star rating), runs 100-round bootstrap for confidence intervals, normalizes severity scores 0–1
4. **Impact table** → `systemic_detector.py` computes coefficient of variation of similarity scores per theme, combines consistency + frequency + impact into weighted composite score, classifies issues
5. **Classified impact data** → `roadmap_generator.py` sends top 5 themes with quantitative context to Groq LLM, receives prioritized JSON recommendations
6. **Full pipeline output** → `evaluation.py` validates theme extraction against 8 hand-labeled test cases, generates similarity distribution and regression diagnostic plots
7. **Everything** → compiled into `results.json` and displayed via Streamlit dashboard

### Why This Architecture?

Each module is **independent, testable, and has a single responsibility**. The pipeline flows from raw data to business-ready output without circular dependencies. This separation means we can swap any component (e.g., replace Ridge with XGBoost, swap the embedding model, switch from Groq to OpenAI) without affecting the rest.

### Trade-offs Considered

| Decision | Choice | Alternative | Rationale |
|---|---|---|---|
| Theme detection | Sentence embeddings | Keyword lists, BERTopic/LDA | Embeddings generalize to unseen phrasings; keyword lists are fragile and indefensible |
| Impact model | Ridge regression | Random Forest, XGBoost | Ridge gives interpretable coefficients ("Wait Time costs -0.2 stars"); Random Forest gives importance but not direction |
| LLM usage | Roadmap only | LLM for theme extraction too | Keeping the analytical core deterministic and reproducible; LLM adds value only for natural language recommendations |
| Systemic detection | Variance-based composite | Simple frequency threshold | Captures whether a problem is *uniformly experienced* (consistency), not just how often it appears |
| Confidence | Bootstrap resampling | Single-point estimates | 100-iteration bootstrap gives coefficient sign stability — much more rigorous than a single R² |

---

## 5. Data Handling & Preprocessing

### Dataset

[Kaggle Hospital Reviews Dataset](https://www.kaggle.com/datasets/junaid6731/hospital-reviews-dataset) — 1,001 reviews with free-text feedback, sentiment labels, and star ratings (1–5).

### Preprocessing Steps

| Step | Action | Detail |
|---|---|---|
| 1 | Column renaming | `Feedback` → `review_text`, `Ratings` → `rating`, `Sentiment Label` → `sentiment_label` |
| 2 | Null removal | Drop rows with null review text |
| 3 | Noise filtering | Drop reviews shorter than 20 characters |
| 4 | Text cleaning | Strip HTML tags, remove URLs, collapse whitespace, lowercase |
| 5 | Rating normalization | Coerce to numeric, clip to 1–5 scale |
| 6 | Feature engineering | Add `review_length` (character count) and `word_count` (token count) |

**Result:** 991 reviews retained out of 1,001 — **99.5% retention rate** (minimal data loss from cleaning).

### EDA Highlights

| Statistic | Value |
|---|---|
| Total reviews | 991 |
| Average rating | 3.56 / 5.0 |
| Negative reviews (1–2 stars) | 27.0% |
| Positive reviews (4–5 stars) | 60.5% |
| Neutral reviews (3 stars) | 12.5% |

### Data Limitations

- **No date column** — trend analysis over time is not possible
- **No hospital name column** — cross-hospital comparison not available
- **Single-source dataset** — may not generalize to other regions or hospital types
- **No multi-language reviews** — all text is English

---

## 6. Modeling & AI Strategy

### Model 1: Theme Extraction — Sentence Embeddings + Cosine Similarity

**Model:** `all-MiniLM-L6-v2` (22M parameters, ~80MB, runs on CPU)

**How it works:**
1. Define each of 9 operational themes as a single short phrase (semantic anchor). For example: `"long waiting time and delays"` for Wait Time, `"staff and nurse behavior and attitude"` for Staff Behavior
2. Encode all 9 theme descriptions and all 991 reviews into the same 384-dimensional embedding space using the sentence transformer
3. Compute cosine similarity matrix of shape (991 × 9) between all review-theme pairs
4. If similarity > 0.3 for a given pair, assign that theme to the review. A review can match multiple themes simultaneously
5. Store raw similarity scores for downstream variance analysis in the systemic detector

**Why this model?**
- Lightweight enough to run on any laptop (no GPU required)
- Pre-trained on 1B+ sentence pairs — strong semantic understanding out of the box
- **Deterministic** — same input always produces same output (unlike LLM-based extraction)
- Handles paraphrasing naturally: "I waited 3 hours" matches "long waiting time and delays" without keyword engineering

**Alternatives considered:**
| Alternative | Why Rejected |
|---|---|
| Keyword matching | Fragile, requires maintaining large keyword lists, misses paraphrases |
| BERTopic / LDA | Unsupervised — themes are unnamed and may not align with operational categories |
| Zero-shot classification (Hugging Face) | Slower, requires larger models; our approach achieves similar results with simpler architecture |
| LLM-based extraction | Non-deterministic, expensive, overkill for a classification task |
| K-Means clustering | Clusters are unlabeled, K is a hyperparameter to guess, one review can only belong to one cluster |

### Model 2: Impact Quantification — Ridge Regression

**Model:** `Ridge(alpha=1.0)` — L2-regularized linear regression

**How it works:**
- **Features:** 9 binary theme columns (detected / not detected)
- **Target:** Star rating (1–5)
- **Output:** Per-theme coefficient — e.g., `-0.208` for Wait Time means reviews mentioning wait time are associated with a 0.2-star rating drop

**Confidence estimation:** Bootstrap resampling (100 iterations). Each round resamples the data with replacement, fits a new Ridge model, and records coefficients. Confidence = proportion of bootstrap samples where the coefficient sign matches the main estimate.

| Theme | Rating Impact | Bootstrap Confidence |
|---|---|---|
| Wait Time | -0.208 stars | 100% |
| Billing & Insurance | -0.192 stars | 100% |
| Appointment Scheduling | -0.134 stars | 100% |
| Communication | +0.146 stars | 100% |
| Staff Behavior | +0.093 stars | 97% |

**Why Ridge?**
- Coefficients are **directly interpretable** as "how much this theme shifts the rating"
- L2 regularization handles correlated themes without overfitting
- Cross-validated R² measures overall model fit
- A hospital manager understands "-0.2 stars" — they don't understand "feature importance = 0.15"

**Alternatives considered:**
| Alternative | Why Rejected |
|---|---|
| Random Forest | Good for importance ranking but coefficients are not interpretable |
| SHAP values | Expensive to compute; Ridge coefficients already give directional impact |
| Pearson correlation | Too simplistic — doesn't control for confounding themes |

### Model 3: Systemic Detection — Variance-Based Composite Scoring

**How it works:**
- For each theme, compute the **coefficient of variation** (CV = std/mean) of cosine similarity scores among reviews where that theme was detected
- Consistency = `1 - CV`. Low CV = theme appears with **uniform strength** across many reviews (systemic). High CV = sporadic spikes in a few reviews (isolated)
- Combine three normalized signals into a composite systemic score:
  - **Consistency:** 40% weight
  - **Frequency:** 30% weight
  - **Absolute rating impact:** 30% weight
- Classification: SYSTEMIC (score ≥ 0.5 AND negative impact) | ISOLATED (score < 0.25) | MODERATE (everything else)

### Model 4: Roadmap Generation — Groq LLM

**Model:** `llama-3.1-8b-instant` via Groq API

**Prompt strategy:** Provide the LLM with *only* the structured impact data (top 5 themes, frequency, rating impact, severity classification) and ask for 5–7 prioritized recommendations as a JSON array. The LLM **never sees raw reviews** — it operates on validated analytics output.

**Grounding method:** Every recommendation references a specific theme with measured impact. The system works fully without the LLM (returns empty roadmap) — the analytical core is deterministic.

**Limitations:**
- LLM recommendations are suggestions, not validated actions
- Quality depends on the model's understanding of healthcare operations
- Runs in demo mode (empty roadmap) without a Groq API key

---

## 7. Evaluation & Metrics

### Theme Extraction Accuracy

**8 hand-crafted test cases** covering all 9 operational themes with known expected outputs.

**Example test cases:**

| # | Test Review | Expected Themes | Detected Themes | Result |
|---|---|---|---|---|
| 1 | "I waited over 3 hours past my appointment. The delay was unacceptable." | wait_time, appointment_scheduling | wait_time, appointment_scheduling | Exact match |
| 2 | "The hospital was filthy. Bathrooms were disgusting and rooms smelled bad." | cleanliness | cleanliness, facility | Partial (extra theme) |
| 3 | "Billing department overcharged my insurance and the food was terrible." | billing, food | billing, food | Exact match |
| 4 | "Nurses were rude and dismissive. No one communicated test results." | staff_behavior, communication | staff_behavior, communication | Exact match |
| 5 | "Parking lot was full and I had to walk 10 minutes in the rain." | parking | parking, facility | Partial (extra theme) |

### Metrics

| Metric | Score | What It Measures |
|---|---|---|
| **Avg Precision** | 0.791 | Of themes detected, 79% were correct |
| **Avg Recall** | 0.666 | Of expected themes, we caught 67% |
| **Avg F1** | 0.709 | Harmonic mean — overall accuracy |
| **Perfect Match Rate** | 50% | Half of test cases got the exact right theme set |

**Why these metrics?**
- Precision measures over-detection (false positives). A hospital manager doesn't want phantom issues.
- Recall measures under-detection (false negatives). We don't want to miss real problems.
- F1 balances both — critical because both over- and under-detection have business consequences.

**Limitations:** Test cases are hand-labeled and may reflect our own biases. A larger, independently labeled test set with inter-annotator agreement would be more rigorous.

### Regression Model Quality

- **Cross-validated R²:** Measures how much of the rating variance is explained by theme presence alone
- **Per-theme bootstrap confidence:** 100 resampling iterations measuring coefficient sign stability (>0.9 = highly confident direction)

### Evaluation Visualizations

The pipeline generates two evaluation plots in `outputs/`:

**1. Similarity Distribution (`similarity_distribution.png`)**

A 3×3 grid of histograms showing the cosine similarity distribution for each of the 9 themes across 200 sampled reviews. Each histogram includes the 0.3 threshold as a red dashed line, visualizing how cleanly the threshold separates matching vs. non-matching reviews.

![Similarity Distribution](outputs/similarity_distribution.png)

**2. Regression Diagnostics (`regression_diagnostics.png`)**

Two-panel figure: (left) Ridge regression coefficients per theme as a horizontal bar chart — red for negative impact, green for positive; (right) bootstrap confidence per theme. Shows which themes reliably help or hurt ratings.

![Regression Diagnostics](outputs/regression_diagnostics.png)

---

## 8. Business Impact & Actionability

### What the output tells a hospital manager

| Question a Manager Asks | ClinsightAI Answer | Evidence |
|---|---|---|
| "What are patients complaining about?" | Theme frequency ranking | Communication: 29%, Staff Behavior: 28.6%, Wait Time: 7.9% |
| "Which complaints actually hurt our ratings?" | Rating impact scores | Wait Time: -0.208 stars, Billing: -0.192 stars |
| "Are these one-off or systemic?" | Issue classification | Wait Time: SYSTEMIC, Staff Behavior: MODERATE |
| "What should we fix first?" | Severity-ranked roadmap | Severity score (frequency × impact, normalized 0–1) + prioritized LLM recommendations |
| "What do 1-star reviews look like vs 5-star?" | Rating segment analysis | Wait Time: 15.7% of 1-star vs 3.5% of 5-star (4.5x gap) |

### Decision-Ready Insights

A hospital COO could open the JSON output or dashboard and immediately see:
- **Wait Time** appears in 15.7% of 1-star reviews vs 3.5% of 5-star reviews → clear operational target
- **Billing** has the strongest negative impact (-0.192 stars) with 100% bootstrap confidence → high priority
- **Communication** is frequent (29%) but has *positive* impact (+0.146) → patients praise it, don't fix what isn't broken

Every recommendation ties back to a quantified theme with measured impact. This is not "improve patient experience." This is **"fix scheduling — it is costing you 0.13 stars with 100% confidence."**

### Real-World Usability

- Output format (JSON) is directly parseable by existing hospital BI systems
- Dashboard requires no technical expertise to navigate
- Roadmap provides specific, prioritized actions with expected rating lift
- System can be re-run on any new dataset — no retraining required

### Limitations

- Analysis is **correlational, not causal** — a theme being associated with low ratings doesn't guarantee fixing it will improve ratings by the exact predicted amount
- Recommendations are AI-generated suggestions requiring domain expert validation
- Dataset is a single snapshot — continuous monitoring would require regular re-runs
- No patient demographic data — cannot segment insights by patient group

---

## 9. Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **NLP / Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) |
| **ML / Statistics** | scikit-learn (Ridge regression, StandardScaler, cross-validation) |
| **LLM** | Groq API (LLaMA 3.1 8B Instant) |
| **Dashboard** | Streamlit + Plotly |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib (evaluation plots), Plotly (dashboard) |
| **Config** | python-dotenv |

---

## 10. How to Run the Project

### Clone Repository
```bash
git clone https://github.com/your-repo/ClinsightAI-Clinsight.git
cd ClinsightAI-Clinsight
```

### Install Dependencies
```bash
conda create -n clinsight-nlp python=3.10
conda activate clinsight-nlp
pip install -r requirements.txt
```

### Run the Analysis Pipeline
```bash
python main.py
```
This generates `outputs/results.json`, `outputs/similarity_distribution.png`, and `outputs/regression_diagnostics.png`.

### Run the Interactive Dashboard
```bash
streamlit run dashboard/app.py
```
Opens a 7-page Streamlit dashboard at `http://localhost:8501`.

### (Optional) Enable LLM Roadmap
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Without a key, the pipeline runs in demo mode — all analytics work, but the LLM-generated roadmap returns empty.

### Custom Dataset
```bash
python main.py --data path/to/your/reviews.csv --output outputs/custom_results.json
```
The CSV must have `Feedback` (text) and `Ratings` (1–5) columns.

---

## 11. Repository Structure

```
ClinsightAI/
├── data/
│   └── hospital.csv                  # Kaggle hospital reviews dataset (1,001 reviews)
├── src/
│   ├── config.py                     # Central config: themes, thresholds, API setup
│   ├── data_loader.py                # CSV loading, text cleaning, feature extraction
│   ├── theme_extractor.py            # Embedding-based semantic theme matching
│   ├── impact_quantifier.py          # Ridge regression + bootstrap confidence intervals
│   ├── systemic_detector.py          # Variance-based SYSTEMIC / MODERATE / ISOLATED classification
│   ├── roadmap_generator.py          # Groq LLM roadmap + executive summary computation
│   └── evaluation.py                 # Test cases, precision/recall/F1, visualization generation
├── dashboard/
│   └── app.py                        # 7-page Streamlit interactive dashboard
├── outputs/
│   ├── results.json                  # Structured JSON pipeline output
│   ├── similarity_distribution.png   # Theme similarity distribution (evaluation)
│   └── regression_diagnostics.png    # Ridge coefficients + bootstrap confidence (evaluation)
├── docs/
│   └── architecture_diagram.png      # System architecture diagram
├── main.py                           # Main pipeline entry point (orchestrates all 7 steps)
├── requirements.txt                  # Python dependencies with version pins
├── DEMO_WALKTHROUGH.md               # Detailed presentation script
└── README.md                         # This file
```

---

## 12. Alignment with HackVerse Rubric

| Rubric Category (Points) | How We Address It | Key Evidence |
|---|---|---|
| **Problem Understanding (10 pts)** | Clear business framing — who the user is (hospital COO), why it matters (0.5-star = patient retention), measurable goals, 5 assumptions stated | Section 1: end user defined, success criteria defined, assumptions enumerated |
| **Data & System Design (15 pts)** | Meaningful EDA (rating distributions, sentiment breakdown, review length analysis), clean 7-stage modular pipeline, architecture diagram, data flow description | Section 4: architecture diagram, trade-off table; Section 5: preprocessing table, EDA stats |
| **Technical Depth (20 pts)** | Goes far beyond sentiment — embedding-based theme matching, regression impact scoring, variance-based systemic classification, 1-star vs 5-star segmentation, composite scoring | Section 6: four distinct models, each with alternatives considered and rejected |
| **Modeling Strategy (20 pts)** | Hybrid approach: semantic embeddings + Ridge regression + variance analysis + LLM. Each model choice explained with alternatives table. Prompt strategy and grounding method documented | Section 6: model-by-model explanation, "why this model" rationale, alternatives tables |
| **Evaluation & Metrics (15 pts)** | 8 test cases, precision/recall/F1 metrics, 2 evaluation visualizations (similarity distribution + regression diagnostics), metric explanations with limitations | Section 7: 5 example test cases shown, metrics table, visualizations embedded, limitations stated |
| **Business Actionability (10 pts)** | Structured JSON matching problem statement format, prioritized recommendations with expected rating lift, decision-ready insights table, real-world usability discussed | Section 8: 5-question manager Q&A table, specific actionable numbers |
| **Visualization & UX (5 pts)** | 7-page Streamlit dashboard: Executive Summary, EDA, Theme Analysis, Impact Matrix, Systemic Issues, Action Roadmap, Raw Data Explorer. Clean Plotly charts, filterable views | Section 11: dashboard described; live demo |
| **Demo & Communication (5 pts)** | Live dashboard demo, clear pipeline execution with progress indicators, DEMO_WALKTHROUGH.md with full presentation script | DEMO_WALKTHROUGH.md: section-by-section script with timing |
| **Bonus: Hybrid retrieval** | Embedding-based semantic matching + statistical regression + variance analysis | Three complementary techniques combined |
| **Bonus: Advanced analytics depth** | Bootstrap confidence intervals, coefficient of variation for consistency, composite scoring with weighted signals | Sections 6–7: statistical rigor beyond basic ML |
| **Bonus: Explainability** | Every prediction backed by coefficient, confidence interval, and evidence samples. No black-box outputs | Entire pipeline designed for interpretability |

---

## Compliance Statement

We confirm that this project was developed during HackVerse 2026.
We used only permitted datasets and tools.
No private code sharing occurred between teams.
All work is original.
