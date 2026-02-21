# ClinsightAI — Presentation Script

---

## [SLIDE / OPEN: Title + Team Info] ~ 15 seconds

Good morning. We are Neural Ninjas, and we built ClinsightAI for the Clinsight company problem statement.

---

## [SECTION 1: PROBLEM UNDERSTANDING] ~ 45 seconds

**Rubric: 10 pts — Problem Understanding & Business Framing**

So here is the problem. Multi-location healthcare groups collect thousands of patient reviews, but this data just sits there. It is unstructured, noisy, and very hard to act on.

Hospital operations managers today cannot answer basic questions like: Which specific operational problems are actually hurting our ratings? Are these complaints systemic or just one-off incidents? And most importantly, what should we fix first?

The problem statement asks us to go beyond basic sentiment analysis. Anyone can count positive and negative reviews. The real challenge is quantifying which operational themes drive ratings, detecting recurring systemic issues, and converting that into an actionable improvement roadmap.

Our end user is a hospital COO or operations manager who needs to make data-driven decisions about where to invest in improvement. Success means they can open our output and immediately know what to fix, in what order, and with what confidence.

---

## [SECTION 2: SOLUTION OVERVIEW] ~ 30 seconds

**Rubric: Transition to architecture**

ClinsightAI is a 7-step pipeline. It takes a hospital review CSV as input, runs embedding-based theme detection, Ridge regression for impact scoring, variance-based systemic classification, and LLM-powered roadmap generation. The output is a structured JSON report that matches the exact format from the problem statement, plus an interactive Streamlit dashboard.

What makes it unique is the hybrid approach. We combine sentence embeddings for theme detection with statistical regression for impact quantification and an LLM only for the final recommendation step. The analytical core is fully deterministic and reproducible.

---

## [SECTION 3: ARCHITECTURE — FILE BY FILE] ~ 60 seconds

**Rubric: 15 pts — Data & System Design**

Let me walk through the architecture file by file. [Show the pipeline diagram.]

Our project has 8 core Python files. I will explain each one.

### `src/config.py` — Central Configuration

This is the single source of truth for the entire pipeline. It defines:

- **9 operational themes** as a list: wait_time, staff_behavior, billing, cleanliness, communication, facility, appointment_scheduling, food, parking.
- **Human-readable labels** for each theme in a dictionary, for example wait_time maps to "Wait Time", billing maps to "Billing & Insurance".
- **Groq API setup**: it loads the API key from a `.env` file using python-dotenv. If no key is found, it sets `MOCK_MODE = True`, which means the pipeline runs without the LLM. The LLM model is `llama-3.1-8b-instant`.

Every other module imports from config.py. This means if we want to add a new theme, we change one file and it propagates everywhere.

### `src/data_loader.py` — Step 1: Load & Preprocess

This file has four functions:

- `load_dataset()` reads the CSV and renames the Kaggle columns to snake_case: `Feedback` becomes `review_text`, `Ratings` becomes `rating`, `Sentiment Label` becomes `sentiment_label`. It also drops any empty unnamed columns.
- `preprocess()` does five things: drops null reviews, drops reviews shorter than 20 characters as noise, runs `clean_text()` on every review, coerces ratings to numeric and clips them to a 1–5 scale, and adds two derived features: `review_length` and `word_count`.
- `clean_text()` is the text cleaning function. It strips HTML tags with a regex, removes URLs, collapses multiple spaces, and lowercases the text.
- `get_eda_stats()` computes summary statistics: total reviews, average rating, rating distribution as a dictionary, average review length, and percentage breakdowns for negative (1–2 stars), positive (4–5 stars), and neutral (3 stars) reviews.

After preprocessing, we go from 1,001 to 991 reviews — 99.5% retention rate, meaning almost no data is lost.

### `src/theme_extractor.py` — Step 2: Embedding-Based Theme Detection

This is where the core NLP happens. I will explain the model in detail in the modeling section, but here is what the file does:

- **Theme definitions**: 9 short phrases, one per theme. For example, wait_time is `"long waiting time and delays"`, staff_behavior is `"staff and nurse behavior and attitude"`. These are semantic anchors, not keyword lists.
- `_load_model()` loads the `all-MiniLM-L6-v2` sentence transformer. 22 million parameters, about 80 megabytes, runs entirely on CPU.
- `run_theme_extraction()` is the main function. It encodes all 9 theme descriptions and all 991 reviews into a shared 384-dimensional embedding space using the sentence transformer. Then it computes a cosine similarity matrix of shape (991 reviews × 9 themes). For each review-theme pair, if similarity exceeds 0.3, we set a boolean flag `theme_{name}` to True, store the similarity as `severity_{name}`, and also store the raw similarity score as `sim_{name}`. The raw scores are used later by the systemic detector for variance analysis. It also assigns an `overall_sentiment` label based on rating: 1–2 is negative, 3 is neutral, 4–5 is positive.
- `get_theme_summary()` aggregates across all reviews: for each theme, it computes frequency count, frequency percentage, average severity, average rating among reviews with that theme, the rating delta versus the overall mean, and picks the 3 lowest-rated reviews as evidence samples.

### `src/impact_quantifier.py` — Step 3: Ridge Regression

This file answers the question: which themes actually hurt or help ratings?

- `run_regression()` builds a Ridge regression model. Features are the 9 binary theme columns. Target is the star rating. It uses `StandardScaler` to normalize features, then fits `Ridge(alpha=1.0)`. It runs 5-fold cross-validation to get the R² score. Then it does 100 rounds of bootstrap resampling: each round resamples the data with replacement, fits a new Ridge model, and records the coefficients. Confidence per theme is the proportion of bootstrap samples where the coefficient sign matches the main estimate. So if Wait Time is negative in 100 out of 100 bootstrap rounds, that is 100% confidence.
- `build_impact_table()` combines the regression output with theme summary data. For each theme, it computes the Ridge coefficient as `rating_impact`, raw severity as `(frequency / 100) × |rating_impact|`, then normalizes severity to a 0–1 scale. It ranks themes by severity score and adds a rank column. The output is the `impact_df` dataframe used by every downstream module.
- `get_rating_segments()` splits reviews into low-rating (1–2 stars) and high-rating (4–5 stars) groups, then computes theme prevalence in each group. This powers the 1-star vs 5-star comparison chart.

### `src/systemic_detector.py` — Step 4: Variance-Based Systemic Detection

This is where we classify whether a theme represents a systemic problem, a moderate concern, or an isolated incident. We use a variance-based approach built on three signals.

- `_compute_consistency()` is the core function. For each theme, it takes the cosine similarity scores among only the reviews where that theme was detected, and computes the coefficient of variation, which is the standard deviation divided by the mean. We then compute consistency as `1 - CV`. A low CV means the theme appears with uniform strength across many reviews — that is a systemic pattern. A high CV means some reviews barely match and others match strongly — that is sporadic and isolated.
- `classify_issues()` combines three normalized signals into a composite systemic score: consistency gets 40% weight, frequency gets 30% weight, and absolute rating impact gets 30% weight. Each signal is normalized to a 0–1 scale. A theme is classified as SYSTEMIC if its composite score is at or above 0.5 AND it has a negative rating impact. ISOLATED if the score is below 0.25. Everything else is MODERATE. This replaces our earlier approach of using a single hard-coded frequency threshold of 12%.
- `cluster_reviews()` labels each individual review as High Risk (1–2 stars), Moderate Risk (3 stars), or Positive Experience (4–5 stars).
- `get_systemic_summary()` aggregates the classifications into lists of systemic, moderate, and isolated themes, and picks the top 3 by escalation score.

### `src/roadmap_generator.py` — Step 5: LLM-Powered Recommendations

This file has two responsibilities: generating the roadmap and computing the executive summary.

- `generate_roadmap()` only runs if a Groq API key is set. It takes the top 5 themes from the impact table — their labels, frequency, rating impact, and severity — and constructs a prompt for the LLM. The prompt tells the model to act as a healthcare operations consultant, provides the hospital summary stats and top issues, and asks for 5–7 prioritized recommendations as a JSON array with priority, recommendation text, expected rating lift, and confidence. The LLM is `llama-3.1-8b-instant` via the Groq API with temperature 0.3 for consistency. Critically, the LLM never sees raw reviews — it only operates on our validated quantitative output.
- `_parse_json()` is a fallback JSON parser. If the LLM wraps the JSON in markdown or extra text, this function finds the JSON array or object by matching brackets and extracting it.
- `get_executive_summary()` computes the health score as `(avg_rating / 5) × 100`, assigns a label (Good/Needs Improvement/Critical), and identifies the top 3 risk themes, which are the themes with negative rating impact sorted by severity.

If no API key is set, the roadmap returns an empty list, but everything else in the pipeline works normally.

### `src/evaluation.py` — Step 6: Evaluation & Metrics

This file validates our theme extraction accuracy and generates two visualizations.

- **Test cases**: 8 hand-crafted reviews with known expected themes covering all 9 themes. For example: "I waited over 3 hours past my appointment. The delay was unacceptable." should detect wait_time and appointment_scheduling.
- `run_theme_test_cases()` re-runs the embedding model on these 8 reviews, detects themes using the same cosine similarity logic, and computes per-test precision, recall, and F1 by comparing detected versus expected themes.
- `compute_overall_metrics()` averages precision, recall, and F1 across all 8 tests, and computes the perfect match rate, which is the proportion of tests where the detected themes exactly match the expected themes.
- `plot_similarity_distribution()` samples 200 reviews, encodes them, and generates a 3×3 grid of histograms showing the cosine similarity distribution for each theme. Each histogram has the 0.3 threshold as a red dashed line. This shows how cleanly the threshold separates matching from non-matching reviews.
- `plot_regression_diagnostics()` generates a two-panel figure. The left panel shows the Ridge coefficient per theme as a horizontal bar chart — red for negative impact, green for positive. The right panel shows bootstrap confidence per theme.

Results: precision 0.791, recall 0.666, F1 0.709, perfect match rate 50%.

### `main.py` — Pipeline Orchestrator

This is the entry point. It runs all 7 steps in sequence:

1. Calls `load_dataset()` and `preprocess()` from data_loader
2. Calls `run_theme_extraction()` and `get_theme_summary()` from theme_extractor
3. Calls `build_impact_table()` from impact_quantifier
4. Calls `classify_issues()` and `cluster_reviews()` from systemic_detector
5. Calls `generate_roadmap()` and `get_executive_summary()` from roadmap_generator
6. Calls `run_full_evaluation()` from evaluation
7. Compiles everything into a single JSON dictionary and writes it to `outputs/results.json`

It accepts two command-line arguments: `--data` for the input CSV path and `--output` for the output JSON path. It prints a summary at the end showing the overall rating, health score, systemic issue count, and top risk theme.

### `dashboard/app.py` — Streamlit Dashboard

This is a 7-page interactive dashboard built with Streamlit and Plotly.

The sidebar has navigation and shows whether the LLM is active or in demo mode.

The 7 pages are:

1. **Executive Summary**: 4 KPI metric cards (total reviews, avg rating, health score, % negative), a color-coded rating distribution bar chart, and top risk themes with severity indicators.
2. **EDA Overview**: rating distribution, sentiment pie chart, review length histogram, rating vs review length box plot, and a 1-star vs 5-star grouped bar chart comparing theme prevalence.
3. **Theme Analysis**: horizontal bar chart of theme frequency colored by severity, plus a deep-dive dropdown that shows frequency, severity, rating impact, and evidence samples for any selected theme.
4. **Impact Matrix**: bubble scatter plot where x is frequency, y is rating impact, size is severity score. Themes below the zero line hurt ratings. Bigger bubbles mean higher severity. Includes a ranked impact data table.
5. **Systemic Issues**: three-column layout showing systemic (red), moderate (yellow), and isolated (green) themes, plus an escalation risk bar chart.
6. **Action Roadmap**: expandable cards for each LLM-generated recommendation with priority, expected rating lift, and confidence. Shows an info message in demo mode.
7. **Raw Data**: filterable table of all 991 reviews. Filter by star rating and detected theme.

The dashboard runs the full analysis pipeline on load using `@st.cache_data` so it only computes once per session.

Each module is independent and testable. We can swap any component without affecting the rest.

---

## [SECTION 4: MODELING STRATEGY — THEME EXTRACTION] ~ 60 seconds

**Rubric: 20 pts — Modeling & AI Strategy (most important section)**

Now the key technical decision. How do we detect themes?

The naive approach is keyword matching. You define a list of words for each theme, like "wait", "delay", "hours" for wait time. The problem is these lists are fragile, they miss paraphrases, and you cannot defend how you chose those exact keywords.

We also considered BERTopic and LDA for unsupervised topic modeling, but those give you unnamed topics that may not align with the operational categories a hospital manager actually cares about.

What we do instead is define each theme as a single short phrase. For example, wait time is defined as "long waiting time and delays." Staff behavior is "staff and nurse behavior and attitude." Just one sentence per theme.

We then use the all-MiniLM-L6-v2 sentence transformer, which is a 22-million parameter model, about 80 megabytes, runs on CPU, no GPU required, to encode both the theme descriptions and all 991 reviews into a 384-dimensional embedding space. We compute cosine similarity between every review and every theme. If the similarity exceeds 0.3, we assign that theme.

This is essentially zero-shot semantic classification. The model has never seen our data during training, but because it was pre-trained on over a billion sentence pairs, it understands that "I waited 3 hours" is semantically close to "long waiting time and delays."

The key advantage is that this approach is defensible, reproducible, and generalizes to unseen phrasings. No keyword maintenance required.

---

## [SECTION 5: MODELING STRATEGY — IMPACT QUANTIFICATION] ~ 45 seconds

**Rubric: 20 pts continued**

Once we have themes assigned, we need to answer: which themes actually hurt ratings?

We use Ridge regression, which is L2-regularized linear regression. The features are 9 binary columns, one per theme, and the target is the star rating. The output is a coefficient per theme. For example, Wait Time has a coefficient of minus 0.208, meaning reviews mentioning wait time are associated with a 0.2-star rating drop.

Why Ridge? Because the coefficients are directly interpretable. A hospital manager can understand "this theme costs you 0.2 stars." Random Forest gives feature importance but not direction. SHAP is expensive and our Ridge coefficients already tell us what we need.

For confidence, we run 100 rounds of bootstrap resampling and measure how consistently each coefficient keeps the same sign. If Wait Time is negative in 100 out of 100 bootstrap samples, that is 100% confidence. Our results show Wait Time, Appointment Scheduling, and Billing all have 100% confidence as negative-impact themes.

Severity score combines frequency and absolute impact, then normalizes to a 0-1 scale. This gives us a single number for prioritization.

---

## [SECTION 6: SYSTEMIC DETECTION + ROADMAP] ~ 45 seconds

**Rubric: 20 pts — Technical Depth**

For systemic detection, we use a variance-based approach instead of a simple frequency threshold. The insight is that a systemic issue is not just frequent — it appears *consistently* across many reviews with uniform detection strength.

We measure this using the coefficient of variation of cosine similarity scores among reviews where each theme was detected. A low CV means the theme triggers with similar strength across many reviews — that is a uniform, systemic pattern. A high CV means it spikes in a few reviews and barely appears in others — that is sporadic and isolated.

We then combine three normalized signals into a composite systemic score: consistency gets 40% weight, frequency gets 30%, and absolute rating impact gets 30%. A theme is classified as SYSTEMIC only if its composite score is at least 0.5 AND it has a negative rating impact. Below 0.25 is ISOLATED. Everything else is MODERATE.

This approach is more principled than a single frequency cutoff because it captures whether the problem is uniform across the patient population, not just how often it appears.

We also do rating-based segmentation. We compare theme prevalence in 1-star reviews versus 5-star reviews. Wait Time appears in 15.7% of 1-star reviews but only 3.5% of 5-star reviews. That 4.5x gap is a clear signal.

For the roadmap, we feed only our structured quantitative results into the Groq LLM, specifically LLaMA 3.1. The LLM never sees raw reviews. It generates prioritized recommendations grounded in our validated data. If no API key is set, the system still works — it just returns an empty roadmap. The core analytics are never dependent on the LLM.

---

## [SECTION 7: EVALUATION] ~ 45 seconds

**Rubric: 15 pts — Evaluation & Metrics**

We have 8 hand-crafted test cases covering all 9 themes. Each test case is a review with known expected themes.

Our results: average precision 0.79, meaning 79% of the themes we detect are correct. Average recall 0.67, we catch two-thirds of the expected themes. Average F1 is 0.71. Perfect match rate is 50%, meaning half our test cases get the exact right set of themes with no extras and no misses.

We also generate two evaluation visualizations. The first is a similarity distribution plot, a 9-panel chart showing the cosine similarity distribution for each theme across 200 sampled reviews. This shows how well our 0.3 threshold separates matching reviews from non-matching ones.

The second is a regression diagnostics plot showing Ridge coefficients with bootstrap confidence per theme. You can visually see which themes reliably help ratings and which reliably hurt them.

We chose precision, recall, and F1 because they measure both whether we over-detect and whether we under-detect. A limitation is that our 8 test cases are hand-labeled, so a larger independently labeled set would be more rigorous.

---

## [SECTION 8: BUSINESS IMPACT] ~ 30 seconds

**Rubric: 10 pts — Business Actionability**

So what can a hospital manager actually do with this?

[Show the JSON output or dashboard.]

They can see: Communication and Staff Behavior are the most frequent themes, but they have positive rating impact. That means patients praise them. Do not fix what is not broken.

Wait Time, Appointment Scheduling, and Billing are the risk themes. They have negative impact with 100% bootstrap confidence. The roadmap gives prioritized recommendations: implement standardized communication protocols for a +0.12 rating lift, deploy staff behavior training for +0.08, optimize scheduling for +0.06.

Every recommendation ties back to a quantified theme with measured impact. This is not "improve patient experience." This is "fix scheduling, it is costing you 0.13 stars with 100% confidence."

---

## [SECTION 9: LIVE DEMO] ~ 30 seconds

**Rubric: 5 pts — Visualization & UX**

[Run `python main.py` or show the Streamlit dashboard.]

The dashboard has 7 pages: Executive Summary, EDA Overview with rating distributions, Theme Analysis, Impact Matrix, Systemic Issues, Action Roadmap, and a Raw Data Explorer.

[Click through 2–3 pages quickly. Show the impact matrix bubble chart and the systemic issues page.]

---

## [CLOSING] ~ 15 seconds

**Rubric: 5 pts — Demo & Communication**

To summarize: ClinsightAI goes beyond sentiment analysis. It uses sentence embeddings for theme detection, Ridge regression for impact quantification, bootstrap resampling for confidence, variance-based consistency analysis for systemic detection, and an LLM for generating grounded recommendations. Every decision is explainable, every metric is validated, and the output is ready for a hospital manager to act on tomorrow.

---

## Q&A — Anticipated Questions

**Q: Why not use BERTopic or LDA?**
A: BERTopic gives unnamed clusters. We need operational categories a hospital manager recognizes, like Wait Time and Billing. Our approach lets us pre-define meaningful themes and use embeddings for matching.

**Q: How did you choose the 0.3 similarity threshold?**
A: We examined the similarity distribution plot. At 0.3, we get a good separation between relevant and irrelevant reviews for most themes. Lowering it increases recall but adds noise. Raising it increases precision but misses paraphrased mentions.

**Q: Why Ridge instead of a more complex model?**
A: We need interpretable coefficients. A hospital manager needs to know "Wait Time costs you 0.2 stars." Ridge gives us that directly. A Random Forest gives importance but not direction or magnitude.

**Q: Is the LLM doing the actual analysis?**
A: No. The LLM only generates natural language recommendations at the very end. All theme detection, impact scoring, and classification are done with deterministic models. The pipeline works without the LLM.

**Q: How does the variance-based systemic detection work?**
A: We compute the coefficient of variation of cosine similarity scores among reviews where each theme was detected. Low variance means the theme appears consistently across many reviews with uniform strength — that is a systemic pattern. We combine this consistency signal with frequency and impact magnitude into a weighted composite score. A theme needs a score of 0.5 or higher AND negative rating impact to be classified as systemic. This is more principled than a single frequency threshold because it captures whether the problem is uniformly experienced by patients, not just how often it is mentioned.

**Q: How would this scale to multiple hospitals?**
A: The pipeline takes any CSV with review text and ratings. You would run it per hospital and compare outputs. The architecture is stateless and modular, so parallelization is straightforward.

**Q: What are the limitations?**
A: Three main ones. First, the analysis is correlational, not causal — fixing a theme may not improve ratings by the exact predicted amount. Second, our 8 test cases are hand-labeled and small. Third, the dataset has no dates, so we cannot track trends over time.
