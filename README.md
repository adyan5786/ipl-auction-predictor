# 🏏 IPL Auction Price Predictor

An academic machine learning project that predicts IPL player auction prices using historical performance statistics and Scikit-learn regression models, deployed as an interactive Streamlit web app.

**Live App:** [IPL Auction Predictor](https://ipl-auction-predictor.streamlit.app/)

---

## 📌 Project Overview

This project estimates an IPL player's likely auction price based on their career batting and bowling statistics, role, team, and auction year. It was built as part of a Computer Engineering ML mini-project (Mumbai University Rev 2019 syllabus), demonstrating core concepts: data preprocessing, categorical encoding, regression modeling, and model evaluation.

---

## 🗂️ Data Sources

Two separate Kaggle datasets were combined to build the final training data:

1. **[IPL Players Statistics](https://www.kaggle.com/datasets/mohammadzamakhan/ipl-players-statistics)** by Mohammad Zama Khan — career batting/bowling stats for 600+ players (runs, wickets, strike rate, economy, etc.). Player names in this dataset use abbreviated format (e.g., `AJ Finch`).

2. **[IPL Player Auction Dataset](https://www.kaggle.com/code/kalilurrahman/ipl-player-auction-data-analysis)** — auction records from 2013–2022 (970 rows, 543 unique players), including sold price, team, role, and year. Player names here use full format (e.g., `Aaron Finch`).

### The Name-Matching Problem

Since the two datasets used different name formats, a direct merge on player name would have failed for most rows. This was solved with a **fuzzy string-matching pipeline** (`merge_data.py`):

1. Exact matches were accepted automatically.
2. Remaining names were matched using `thefuzz`'s `token_sort_ratio` scoring.
3. Matches scoring ≥90 were accepted as confident.
4. Matches scoring 60–89 were flagged for **manual human review** (`matches_to_review.csv`) — each borderline pair was manually verified before being included, and known false-positive matches (e.g., "Aman Khan" incorrectly matched to "Kamran Khan") were identified and corrected.
5. Only manually approved and high-confidence matches were merged into the final dataset (`ipl_merged.csv`).

This human-in-the-loop validation step was a deliberate design choice — automated fuzzy matching alone is not reliable enough for a dataset where similar surnames could otherwise mismerge two different real players.

---

## 🧹 Data Preprocessing

- Missing numeric values filled with column mean; missing categorical values filled with mode.
- Categorical columns (`Role`, `Team`, `Player Origin`) encoded using `LabelEncoder`.
- **Feature leakage fix:** an initial version of the pipeline accidentally included `Unnamed: 0` (a meaningless CSV row index) and the label-encoded `player` identity as model features. This caused the model to partially memorize player identity/row order instead of learning from actual performance stats. Both columns were identified during a code audit and removed — this improved Linear Regression R² from 0.1487 to 0.1724.

---

## 🤖 Model Training

Two regression models were trained and compared, as required by the project's academic scope (no deep learning or ensemble methods):

| Model | MAE (₹) | R² Score |
|---|---|---|
| **Linear Regression** ✅ (selected) | ~1.82 Cr | **0.1724** |
| Decision Tree (max_depth=5) | ~2.03 Cr | -0.5173 |

**Linear Regression was selected** as the better-performing model based on R² score.

### Why R² is modest — and why that's expected

An R² of 0.17 means the model explains roughly 17% of the variance in auction price. This is a realistic and well-documented outcome for this problem domain: **IPL auction price is driven heavily by factors outside any performance-stats dataset** — player reputation, recent form/momentum, franchise-specific bidding strategy, and market hype. Two players with near-identical statistics can sell for very different prices for reasons no CSV captures.

### Investigated but rejected: Year-based price normalization

Since auction prices have visibly inflated over 2013–2022 (average price rose from ~₹1.5 Cr in 2013–2017 to ~₹2.8–3.2 Cr in 2018–2022), a year-normalization approach was tested — rescaling each year's prices to a common baseline before training. This was **tested and rejected**: it *decreased* Linear Regression's R² (0.1724 → 0.0584), indicating that raw Year actually carries useful signal (e.g., correlating with player recency/relevance) that normalization discarded. This experiment is retained in `model_training.py` (commented out) for transparency, and demonstrates that not every reasonable-sounding fix improves a model — a key ML lesson.

### Sensitivity analysis on the "Year" feature

Manual testing revealed that varying only the Auction Year slider (holding all other inputs fixed) could swing the predicted price substantially. A diagnostic check was run to investigate:

- **Multicollinearity check:** Year's correlation with all other features was weak (max |r| = 0.14), ruling out multicollinearity as the cause.
- **Coefficient check:** Year's regression coefficient (~₹76.7 lakh/year) was **not** anomalously large — it ranked mid-pack among all feature coefficients.
- **Conclusion:** The large swing is simply the coefficient applied across Year's full 9-year range (2013–2022), a normal characteristic of linear extrapolation, made more visible by the model's modest overall R². This is documented as a known limitation rather than a bug.

---

## 🖥️ Streamlit App

`app.py` provides an interactive interface where a user inputs a hypothetical player's role, team, origin, auction year, and batting/bowling stats via sidebar controls, and receives a predicted auction price in ₹ Crore.

- Categorical dropdowns are populated dynamically from the trained `LabelEncoder` classes (never hardcoded).
- Numeric inputs use realistic per-stat ranges (e.g., batting average 0–60, bowling economy 0–15).
- Feature scaling (`StandardScaler`) is applied to numeric inputs before prediction, matching the training pipeline.

---

## 📁 Project Structure

```
ipl-auction-predictor/
│
├── data/
│   ├── ipl_stats.csv              # Raw player statistics (Kaggle)
│   ├── ipl_auction.csv            # Raw auction records (Kaggle)
│   ├── ipl_merged.csv             # Final merged dataset (output of merge_data.py)
│   ├── matches_confident.csv      # High-confidence fuzzy matches (score ≥90)
│   └── matches_to_review.csv      # Manually reviewed borderline matches
│
├── model/
│   ├── model.pkl                  # Trained Linear Regression model
│   ├── label_encoders.pkl         # Fitted LabelEncoders for categorical features
│   ├── feature_columns.pkl        # Exact feature column order used in training
│   └── scaler.pkl                 # Fitted StandardScaler for numeric features
│
├── merge_data.py                  # Fuzzy name-matching + dataset merge pipeline
├── model_training.py              # Preprocessing, training, evaluation, export
├── app.py                         # Streamlit frontend
├── requirements.txt
└── README.md
```

---

## ⚙️ Running Locally

```bash
# Clone the repo
git clone https://github.com/adyan5786/ipl-auction-predictor.git
cd ipl-auction-predictor

# Set up virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# (Optional) Re-run the full pipeline from raw data
python merge_data.py
python model_training.py

# Launch the app
streamlit run app.py
```

---

## 🎓 Key Learning Outcomes

- Real-world data integration across inconsistently formatted sources (fuzzy matching + human validation)
- Identifying and correcting feature leakage (identifier columns masquerading as predictive features)
- Comparative model evaluation (Linear Regression vs. Decision Tree) and interpreting R²/MAE
- Diagnosing model behavior through coefficient analysis and correlation checks, rather than blindly trusting predictions
- Recognizing that not all data problems have a fix that improves the model — and documenting a rejected hypothesis is as valuable as a successful one

---

## ⚠️ Disclaimer

This is an academic project. Predictions are estimates based on historical (2013–2022) data and should not be interpreted as real auction price guarantees.