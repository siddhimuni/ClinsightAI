# ClinsightAI
**AI-Driven Healthcare Review Intelligence System**

Automatically extracts structured operational insights from hospital reviews,
quantifies rating impact, detects systemic issues, and generates a financial
improvement roadmap — powered by Claude AI.

---

## Quick Start

### Step 1 — Add API Key (optional but recommended)
1. Go to https://console.anthropic.com → API Keys → Create Key
2. Open `.env` and paste your key:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
   Without a key, the system runs in **Demo Mode** (rule-based AI simulation).

### Step 2 — Add the Dataset
1. Download from Kaggle: https://www.kaggle.com/datasets/junaid6731/hospital-reviews-dataset
2. Place the CSV at: `data/hospital_reviews.csv`

   Without the dataset, sample data is auto-generated for demo purposes.

### Step 3 — Run the Dashboard
```bash
# Activate virtual environment
source ../venv/Scripts/activate       # Windows
# source ../venv/bin/activate         # Mac/Linux

# Launch dashboard
streamlit run dashboard/app.py
```
Open http://localhost:8501 in your browser.

### Step 4 — Run CLI Pipeline (generates JSON output)
```bash
python main.py
# With custom parameters:
python main.py --data data/hospital_reviews.csv --patients 800 --ltv 2500
```

---

## Project Structure

```
ClinsightAI/
├── data/
│   └── hospital_reviews.csv     ← place Kaggle dataset here
├── src/
│   ├── config.py                ← settings, theme taxonomy, thresholds
│   ├── data_loader.py           ← Phase 1&2: ingestion + preprocessing
│   ├── theme_extractor.py       ← Phase 3: dual-engine theme extraction
│   ├── impact_quantifier.py     ← Phase 4: SHAP + regression analysis
│   ├── systemic_detector.py     ← Phase 5: systemic vs isolated issues
│   ├── financial_simulator.py   ← Phase 6: Rating-to-Revenue Bridge (USP)
│   └── roadmap_generator.py     ← Phase 7: Claude-powered action roadmap
├── dashboard/
│   └── app.py                   ← Streamlit 7-page dashboard
├── outputs/
│   └── results.json             ← generated analysis output
├── main.py                      ← CLI pipeline runner
├── requirements.txt
└── .env                         ← API key goes here
```

---

## Architecture

```
Raw Reviews (CSV)
      ↓
[Preprocessing] — clean text, normalize ratings
      ↓
[Dual-Engine Theme Extraction]
   ├── Statistical (keyword matching)
   └── Claude AI (structured JSON extraction)
      ↓
[Impact Quantification] — Ridge regression + SHAP
      ↓
[Systemic Issue Detection] — escalation scoring + trend analysis
      ↓
[Financial Simulator] — Rating → Retention → Revenue (USP)
      ↓
[Roadmap Generator] — Claude-powered action plan
      ↓
[Streamlit Dashboard] — 7-page interactive UI
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| Executive Summary | KPIs, health score, revenue opportunity |
| Theme Analysis | Frequency, severity, evidence quotes |
| Impact Matrix | Bubble chart — frequency vs. rating impact |
| Systemic Issues | Classification, trends, causal chains |
| Financial Simulator | What-If revenue calculator (USP) |
| Action Roadmap | Prioritized improvements with KPIs |
| Raw Data | Filterable review table with theme tags |

---

## Output JSON Format

```json
{
  "clinic_summary": {
    "overall_rating_mean": 3.2,
    "health_score": 64,
    "primary_risk_themes": ["Wait Time", "Billing & Insurance"],
    "revenue_opportunity_annual": 145000
  },
  "theme_analysis": [...],
  "financial_simulation": {...},
  "improvement_roadmap": {
    "quick_wins": [...],
    "strategic": [...],
    "long_term": [...]
  }
}
```
