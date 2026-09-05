# RiskGuard AI — Intelligent Payment Risk Manager

**Detect risk. Explain decisions. Protect transactions.**

Built for the **Razorpay AI Builder Internship 2026 — Track 2: AI Risk Manager**.

> **Disclaimer:** This is a student-built educational prototype. It is **not** an
> official Razorpay product, is not connected to any real bank or payment system,
> and its outputs are not real financial, banking, or regulatory decisions.

---

## 1. Problem Statement

Digital payment platforms process enormous volumes of transactions, a small fraction
of which are fraudulent. Reviewing every transaction manually doesn't scale, and
static rule-based systems miss evolving fraud patterns while generating excessive
false positives that frustrate legitimate customers. Payment platforms need a system
that can score transaction risk in real time, explain *why* a transaction looks
risky, and route only genuinely suspicious activity to human reviewers.

## 2. Why Payment Risk Management Matters

Every fraudulent transaction that gets through is a direct financial loss and an
erosion of customer trust. Every false alarm on a legitimate transaction is friction
that can drive customers away. A good risk system has to balance **recall** (catching
real fraud) against **precision** (not crying wolf on legitimate transactions) —
which is exactly what this project is built to measure and optimize.

## 3. Solution

RiskGuard AI is an end-to-end ML pipeline plus a Streamlit dashboard that:

1. Takes a transaction's raw attributes (type, amount, sender/receiver balances).
2. Computes a **fraud probability** from a trained model.
3. Converts that probability into a **LOW / MEDIUM / HIGH** risk level.
4. Explains the **top contributing factors** using SHAP.
5. Recommends an **action** (allow / verify / manual review).
6. Supports both single-transaction analysis and CSV batch analysis.

## 4. Dataset

**Source:** Kaggle — "Online Payment Fraud Detection" (`onlinefraud.csv`), a
PaySim-style simulated mobile-money transaction log.

Raw schema (as actually found in the file, verified by direct inspection — nothing assumed):

| Column | Type | Description |
|---|---|---|
| `step` | int | Time step of the simulation (1 hour per step, range 1–743) |
| `type` | category | Transaction type: `PAYMENT`, `CASH_OUT`, `CASH_IN`, `TRANSFER`, `DEBIT` |
| `amount` | float | Transaction amount |
| `nameOrig` | string | Sender ID |
| `oldbalanceOrg` | float | Sender balance before the transaction |
| `newbalanceOrig` | float | Sender balance after the transaction |
| `nameDest` | string | Recipient ID |
| `oldbalanceDest` | float | Recipient balance before the transaction |
| `newbalanceDest` | float | Recipient balance after the transaction |
| `isFraud` | int | **Target.** 1 = fraudulent transaction |
| `isFlaggedFraud` | int | A simple rule-based flag from the data source (not a model feature — see §6) |

**Actual dataset facts** (computed directly from the file):

- **6,362,620** rows, **11** columns
- **0** missing values, **0** duplicate rows
- **8,213** fraud transactions out of 6,362,620 → fraud rate **0.1291%**
- Fraud occurs **only** in `TRANSFER` and `CASH_OUT` transactions — 0 fraud cases in
  `PAYMENT`, `CASH_IN`, or `DEBIT` across 3.5M+ such rows
- `nameOrig` has 6,353,307 unique values, `nameDest` has 2,722,362 — both effectively
  unique identifiers
- `isFlaggedFraud` is 1 for only 16 of 6,362,620 rows

## 5. Data Preprocessing & Column Decisions

| Column | Decision | Reason |
|---|---|---|
| `nameOrig`, `nameDest` | **Dropped** | Near-unique identifier strings; would not generalize to unseen customers and risks ID-memorization instead of learning fraud patterns. |
| `isFlaggedFraud` | **Dropped as a feature** | Derived from a hard-coded business rule and only ever 1 for 16/6.36M rows; keeping it as an input would leak rule-based label information. |
| `PAYMENT`, `CASH_IN`, `DEBIT` rows | **Excluded from modeling** | 0 observed fraud cases across 3.5M+ rows of these types — confirmed by direct inspection, not assumed. This also makes training tractable on a normal laptop (2,770,409 rows instead of 6,362,620) without discarding a single fraud example. |
| `step`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `type` | **Kept** | Directly informative, no leakage risk. |

**Engineered features** (derived only from the raw numeric columns above):

```
errorBalanceOrig = oldbalanceOrg - amount - newbalanceOrig
errorBalanceDest = oldbalanceDest + amount - newbalanceDest
```

These capture how far a transaction's before/after balances deviate from simple
bookkeeping — and turned out to be, by far, the strongest real signal in the data
(see §9, Explainable AI).

## 6. Data Leakage Prevention

- All preprocessing (`StandardScaler`, `OneHotEncoder`) is fit **only** on the
  training split, then applied to the test split — never the reverse.
- The test split (20% of the fraud-eligible data, stratified) is **never touched**
  during training, including during the majority-class undersampling step (that
  undersampling happens on the training split only).
- `isFlaggedFraud` and the ID columns are excluded for the leakage reasons above.
- A fixed random seed (`42`) is used for the split, the undersampling, and every
  model, for reproducibility.

## 7. Machine Learning Pipeline

```
Raw CSV (6,362,620 rows)
  → Filter to TRANSFER + CASH_OUT (2,770,409 rows, fraud rate 0.2965%)
  → Feature engineering (errorBalanceOrig, errorBalanceDest)
  → Stratified train/test split (80/20)
       Train: 2,216,327 rows (6,570 fraud)
       Test:    554,082 rows (1,643 fraud) — untouched, real distribution
  → Train-side undersampling of legitimate class to 1:20 fraud:legit ratio
       → 137,970 balanced training rows
  → Preprocessing: StandardScaler (numeric) + OneHotEncoder (type), fit on train only
  → Train & compare 3 models
  → Evaluate all 3 on the untouched test set
  → Select best model by PR-AUC on the fraud class
  → Persist winning pipeline (models/risk_model.joblib)
```

### Models tested

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Train time |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.0433 | 0.8837 | 0.0826 | 0.9759 | 0.5879 | 0.3s |
| Random Forest | 0.9838 | 0.9970 | 0.9903 | 0.9992 | 0.9978 | 32.2s |
| **HistGradientBoosting (selected)** | **0.9951** | **0.9970** | **0.9960** | **0.9994** | **0.9981** | 1.0s |

*(All numbers above are real results from running `src/train_model.py` on this
dataset — see `outputs/training_summary.json` for the full machine-readable results.)*

### Model selection

**HistGradientBoostingClassifier** was selected: it has the highest PR-AUC (the most
informative metric for a ~0.3% fraud rate), matches Random Forest on recall while
having noticeably higher precision, and trains in ~1 second versus 32 seconds for
Random Forest — a meaningful practical advantage on a normal laptop.

The Logistic Regression baseline illustrates why accuracy/ROC-AUC alone can be
misleading here: despite 97.6% ROC-AUC, its precision is only 4.3% — it would flag
huge numbers of legitimate transactions as fraud.

### Confusion matrix (HistGradientBoosting, test set, 554,082 rows / 1,643 fraud)

| | Predicted Legitimate | Predicted Fraud |
|---|---|---|
| **Actual Legitimate** | 552,431 | 8 |
| **Actual Fraud** | 5 | 1,638 |

- **Accuracy:** 0.99998
- **Precision (fraud):** 0.9951
- **Recall (fraud):** 0.9970
- **F1 (fraud):** 0.9960
- **ROC-AUC:** 0.9994
- **PR-AUC (Average Precision):** 0.9981

**Why recall and precision matter more than accuracy here:** with fraud at only
0.30% of eligible transactions, a model that always predicts "legitimate" would
already score 99.7% accuracy while catching zero fraud. Recall tells us how much
real fraud we catch (99.70% here); precision tells us how often a fraud alert is
actually correct (99.51% here) — both need to be high for the system to be useful.

## 8. Explainable AI

Per-transaction explanations use **SHAP `TreeExplainer`**, which computes exact
Shapley values for tree ensembles like HistGradientBoosting — fast enough to run
live in the dashboard for a single transaction, with no approximation needed.

Global feature importance (mean |SHAP value| over a 5,000-row sample of the
modeling dataset):

| Feature | Mean |SHAP value| |
|---|---|
| `errorBalanceOrig` | 1.596 |
| `newbalanceOrig` | 0.998 |
| `oldbalanceOrg` | 0.205 |
| `oldbalanceDest` | 0.183 |
| `step` | 0.152 |
| `amount` | 0.056 |
| `newbalanceDest` | 0.044 |
| `errorBalanceDest` | 0.039 |
| `type = CASH_OUT` | 0.016 |
| `type = TRANSFER` | 0.0005 |

The engineered `errorBalanceOrig` feature dominates: fraudulent transactions in this
dataset tend to leave the sender's balance inconsistent with a simple
before − amount = after check, which is exactly the pattern this feature captures.

## 9. Risk Scoring Methodology

The model outputs a fraud probability in [0, 1]. This is mapped to a risk level
using two **configurable** thresholds (defaults shown; adjustable from the sidebar
on the Transaction Analyzer page):

```
probability <  0.10             → LOW RISK     → Allow transaction
0.10 ≤ probability < 0.50        → MEDIUM RISK  → Request additional verification
probability ≥ 0.50               → HIGH RISK    → Send for manual review
```

These thresholds are prototype defaults for demonstration, not a calibrated,
production risk policy.

## 10. Technology Stack

- Python 3.10+
- pandas, NumPy — data handling
- scikit-learn — preprocessing pipeline, Logistic Regression, Random Forest,
  HistGradientBoostingClassifier
- SHAP — explainability
- Plotly — interactive charts
- Streamlit — web dashboard
- joblib — model persistence

## 11. Project Structure

```
RiskGuard-AI/
├── app.py                      # Streamlit dashboard (all pages)
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── dataset.csv             # place the Kaggle CSV here (gitignored)
├── models/
│   └── risk_model.joblib       # trained pipeline (generated, gitignored)
├── src/
│   ├── data_preprocessing.py   # loading, cleaning, feature engineering
│   ├── train_model.py          # full training + evaluation pipeline
│   ├── prediction.py           # risk scoring, thresholds, recommended actions
│   ├── evaluation.py           # loads real metrics for the app
│   └── explainability.py       # SHAP-based per-transaction explanations
└── outputs/
    └── training_summary.json   # real metrics, confusion matrix, feature importance
```

## 12. Installation

```bash
git clone <your-repo-url>
cd RiskGuard-AI
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the "Online Payment Fraud Detection" dataset from Kaggle and place the CSV
at `data/dataset.csv` (the file is not committed to git because of its size, ~470MB).

## 13. How to Run

**Step 1 — train the model** (only needed once, or after changing the pipeline):

```bash
python src/train_model.py --data data/dataset.csv
```

This trains all three candidate models, evaluates them on a held-out test set,
saves the winning pipeline to `models/risk_model.joblib`, and writes real metrics
to `outputs/training_summary.json`.

**Step 2 — launch the dashboard:**

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## 14. How to Use

- **Dashboard** — overview KPIs and charts computed from the real dataset.
- **Transaction Analyzer** — enter a transaction's details and get a real-time
  risk score, risk level, prediction, and recommended action.
- **Why is this transaction risky?** — SHAP breakdown of the last analyzed transaction.
- **Batch Risk Analysis** — upload a CSV of transactions, get scored results with a
  downloadable risk report.
- **Risk Review Queue** — high-risk transactions from the last batch run, for
  human-in-the-loop review.
- **Model Performance** — real evaluation metrics, confusion matrix, ROC and
  precision-recall curves, model comparison table.
- **Data & Model Insights** — dataset facts, class-imbalance handling, global
  feature importance, and column-exclusion rationale.
- **About** — project explanation and disclaimer.

## 15. Example Workflow

1. Open the Dashboard to see overall fraud statistics.
2. Go to Transaction Analyzer, enter a `TRANSFER` of 181 with sender balance
   181 → 0 and recipient balance 0 → 0 — a pattern resembling the dataset's
   real fraud cases.
3. Click **Analyze Risk** — the model returns a high fraud probability and a
   HIGH risk level.
4. Check **Why is this transaction risky?** to see that `errorBalanceOrig` and
   `newbalanceOrig` are the dominant contributing factors.
5. Upload a batch CSV on **Batch Risk Analysis** and download the scored report.
6. Review flagged transactions in the **Risk Review Queue**.
7. Check **Model Performance** to see the real precision/recall/F1/ROC-AUC/PR-AUC
   backing the system.

## 16. Limitations

- Trained on a single simulated dataset; real payment platforms have richer,
  differently-distributed features and fraud patterns.
- Only covers `TRANSFER` and `CASH_OUT` transactions, since the dataset contains
  no fraud examples of any other type.
- Risk thresholds are illustrative defaults, not calibrated to a real cost-of-fraud
  or regulatory framework.
- Does not connect to any real bank, card network, or Razorpay system.
- Near-perfect metrics reflect this dataset's strong, simulator-generated fraud
  signal (see §8); real-world fraud is typically noisier and harder to separate.

## 17. Future Improvements

- Recalibrate thresholds against a real cost-of-false-positive vs. cost-of-fraud model.
- Add a time-series / velocity feature set (e.g., transactions per account per hour).
- Add model monitoring for data/concept drift once deployed on live traffic.
- Add authentication and audit logging to the Risk Review Queue for real deployment.
- Explore anomaly-detection models (e.g., Isolation Forest) as a second, unsupervised
  signal to catch novel fraud patterns the supervised model hasn't seen.

## 18. Disclaimer

RiskGuard AI is a prototype built for the Razorpay AI Builder Internship 2026,
Track 2: AI Risk Manager, for educational and demonstration purposes only. It is
**not** an official Razorpay product, is not affiliated with or endorsed by
Razorpay, and must not be used to make real financial, banking, or regulatory
decisions.
