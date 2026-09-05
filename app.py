"""
RiskGuard AI — Intelligent Payment Risk Manager
=================================================
Student-built prototype for the Razorpay AI Builder Internship 2026,
Track 2: AI Risk Manager.

Run with:
    streamlit run app.py

This is NOT an official Razorpay product.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from data_preprocessing import (
    ALL_MODEL_FEATURES,
    FRAUD_ELIGIBLE_TYPES,
    TARGET,
    build_feature_row,
    load_raw_dataset,
    prepare_model_frame,
    profile_dataset,
    validate_input_row,
)
from evaluation import get_best_model_metrics, get_model_comparison_table, load_training_summary
from explainability import explain_row, get_tree_explainer
from prediction import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    load_model,
    recommended_action_for,
    score_batch,
    score_to_risk_level,
    score_transaction,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "risk_model.joblib"
SUMMARY_PATH = BASE_DIR / "outputs" / "training_summary.json"

st.set_page_config(
    page_title="RiskGuard AI — Payment Risk Manager",
    page_icon="🛡️",
    layout="wide",
)

RISK_COLORS = {"LOW": "#1a9850", "MEDIUM": "#f5a623", "HIGH": "#d73027"}


# ----------------------------------------------------------------------------
# Cached loaders — the dataset and model are loaded / trained ONCE per process
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def get_raw_dataset():
    return load_raw_dataset(str(DATA_PATH))


@st.cache_data(show_spinner="Preparing modeling data...")
def get_model_frame(_raw_df):
    return prepare_model_frame(_raw_df)


@st.cache_data(show_spinner="Profiling dataset...")
def get_profile(_raw_df):
    return profile_dataset(_raw_df)


@st.cache_resource(show_spinner="Loading trained model...")
def get_model():
    return load_model(str(MODEL_PATH))


@st.cache_resource(show_spinner="Loading explainer...")
def get_explainer(_pipeline):
    return get_tree_explainer(_pipeline)


@st.cache_data(show_spinner=False)
def get_summary():
    return load_training_summary(str(SUMMARY_PATH))


def data_available() -> bool:
    return DATA_PATH.exists()


def model_available() -> bool:
    return MODEL_PATH.exists() and SUMMARY_PATH.exists()


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🛡️ RiskGuard AI")
st.sidebar.caption("Track 2 — AI Risk Manager")

PAGES = [
    "Dashboard",
    "Transaction Analyzer",
    "Why is this transaction risky?",
    "Batch Risk Analysis",
    "Risk Review Queue",
    "Model Performance",
    "Data & Model Insights",
    "About",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption(
    "Built for the Razorpay AI Builder Internship 2026. "
    "This is an educational prototype, not an official Razorpay system."
)

if not data_available():
    st.error(
        f"Dataset not found at `data/dataset.csv`. Place the "
        f"Online Payment Fraud Detection CSV there and reload."
    )
    st.stop()

if not model_available():
    st.warning(
        "No trained model found. Run `python src/train_model.py --data data/dataset.csv` "
        "from the project root first, then reload this app."
    )
    st.stop()

raw_df = get_raw_dataset()
model_df = get_model_frame(raw_df)
profile = get_profile(raw_df)
pipeline = get_model()
explainer = get_explainer(pipeline)
summary = get_summary()

# Session-state thresholds (configurable from the sidebar on the Analyzer page)
if "low_threshold" not in st.session_state:
    st.session_state.low_threshold = DEFAULT_LOW_THRESHOLD
if "high_threshold" not in st.session_state:
    st.session_state.high_threshold = DEFAULT_HIGH_THRESHOLD


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
if page == "Dashboard":
    st.title("RiskGuard AI")
    st.subheader("Intelligent Payment Risk Management")
    st.write(
        "AI-powered transaction risk assessment for detecting potentially "
        "fraudulent payment activity."
    )

    total_txn = profile["n_rows"]
    fraud_txn = profile["fraud_count"]
    fraud_rate = profile["fraud_rate"]
    total_amount = raw_df["amount"].sum()

    # High-risk transactions & amount-at-risk are computed from real model
    # scores on the fraud-eligible subset (scoring all 6.36M rows live on
    # every page load would be slow; we score the same held-out-style sample
    # used throughout the app and note this clearly).
    with st.spinner("Scoring dataset sample for dashboard KPIs..."):
        sample_n = min(200_000, len(model_df))
        scored_sample = model_df.sample(sample_n, random_state=42)
        scored = score_batch(
            pipeline,
            scored_sample[ALL_MODEL_FEATURES],
            st.session_state.low_threshold,
            st.session_state.high_threshold,
        )
        scored["amount"] = scored_sample["amount"].values
        scored["isFraud"] = scored_sample[TARGET].values
        high_risk_share = (scored["risk_level"] == "HIGH").mean()
        amount_at_risk_share = scored.loc[scored["risk_level"] == "HIGH", "amount"].sum() / scored["amount"].sum()

    est_high_risk_txn = int(round(total_txn * high_risk_share))
    est_amount_at_risk = total_amount * amount_at_risk_share

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{total_txn:,}")
    c2.metric("Fraudulent Transactions", f"{fraud_txn:,}")
    c3.metric("Fraud Rate", f"{fraud_rate:.3%}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Total Transaction Value", f"${total_amount:,.0f}")
    c5.metric("High-Risk Transactions (est.)", f"{est_high_risk_txn:,}")
    c6.metric("Amount at Risk (est.)", f"${est_amount_at_risk:,.0f}")

    st.caption(
        "High-risk transaction count and amount-at-risk are estimated by applying the "
        f"trained model to a random {sample_n:,}-row sample of fraud-eligible transactions "
        "(TRANSFER/CASH_OUT) and projecting the observed high-risk share onto the full dataset — "
        "scoring all 6.36M rows on every page load would be impractical for a live dashboard."
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fraud vs Legitimate Transactions")
        counts = raw_df[TARGET].value_counts().rename({0: "Legitimate", 1: "Fraudulent"})
        fig = px.pie(values=counts.values, names=counts.index,
                     color=counts.index,
                     color_discrete_map={"Legitimate": "#1a9850", "Fraudulent": "#d73027"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk Level Distribution")
        risk_counts = scored["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
        fig = px.bar(x=risk_counts.index, y=risk_counts.values,
                     color=risk_counts.index, color_discrete_map=RISK_COLORS,
                     labels={"x": "Risk Level", "y": "Transactions (sample)"})
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Transaction Amount Distribution")
        fig = px.histogram(raw_df.sample(min(100_000, len(raw_df)), random_state=1),
                            x="amount", nbins=60, log_y=True)
        fig.update_layout(xaxis_title="Amount", yaxis_title="Count (log scale)")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Fraud Rate by Transaction Type")
        fbt = raw_df.groupby("type", observed=True)[TARGET].mean().sort_values(ascending=False)
        fig = px.bar(x=fbt.index, y=fbt.values, labels={"x": "Transaction Type", "y": "Fraud Rate"})
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Fraud in this dataset occurs exclusively in **TRANSFER** and **CASH_OUT** "
        "transactions — 0 fraud cases were found in PAYMENT, CASH_IN or DEBIT out of "
        "over 3.5 million such transactions. The model is trained only on these two types."
    )


# ============================================================================
# PAGE: TRANSACTION ANALYZER
# ============================================================================
elif page == "Transaction Analyzer":
    st.title("Transaction Risk Analyzer")
    st.write("Enter transaction details to get a real-time AI risk assessment.")

    with st.sidebar:
        st.divider()
        st.subheader("Risk Thresholds")
        low_t = st.slider("Low → Medium threshold", 0.0, 1.0, st.session_state.low_threshold, 0.01)
        high_t = st.slider("Medium → High threshold", 0.0, 1.0, st.session_state.high_threshold, 0.01)
        if high_t <= low_t:
            st.sidebar.error("High threshold must be greater than low threshold.")
        else:
            st.session_state.low_threshold = low_t
            st.session_state.high_threshold = high_t

    with st.form("txn_form"):
        col1, col2 = st.columns(2)
        with col1:
            txn_type = st.selectbox("Transaction Type", FRAUD_ELIGIBLE_TYPES,
                                     help="Model is trained only on TRANSFER and CASH_OUT — "
                                          "the only types with observed fraud in this dataset.")
            amount = st.number_input("Amount", min_value=0.0, value=5000.0, step=100.0)
            step = st.number_input("Time Step (hour index in simulation)", min_value=1, value=1, step=1)
        with col2:
            oldbalanceOrg = st.number_input("Sender balance before transaction", min_value=0.0, value=10000.0, step=100.0)
            newbalanceOrig = st.number_input("Sender balance after transaction", min_value=0.0, value=5000.0, step=100.0)
            oldbalanceDest = st.number_input("Recipient balance before transaction", min_value=0.0, value=0.0, step=100.0)
            newbalanceDest = st.number_input("Recipient balance after transaction", min_value=0.0, value=5000.0, step=100.0)

        submitted = st.form_submit_button("Analyze Risk", use_container_width=True)

    if submitted:
        raw_input = {
            "type": txn_type, "amount": amount, "step": step,
            "oldbalanceOrg": oldbalanceOrg, "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest, "newbalanceDest": newbalanceDest,
        }
        errors = validate_input_row(raw_input)
        if errors:
            for e in errors:
                st.error(e)
        else:
            feature_row = build_feature_row(
                amount=amount, oldbalanceOrg=oldbalanceOrg, newbalanceOrig=newbalanceOrig,
                oldbalanceDest=oldbalanceDest, newbalanceDest=newbalanceDest,
                type_=txn_type, step=step,
            )
            result = score_transaction(pipeline, feature_row,
                                        st.session_state.low_threshold,
                                        st.session_state.high_threshold)
            st.session_state["last_feature_row"] = feature_row
            st.session_state["last_result"] = result

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        st.divider()
        color = RISK_COLORS[result.risk_level]
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Score", f"{result.fraud_probability:.1%}")
        c2.markdown(
            f"<h4>Risk Level</h4><h2 style='color:{color}'>{result.risk_level} RISK</h2>",
            unsafe_allow_html=True,
        )
        c3.metric("Prediction", result.prediction_label)
        st.info(f"**Recommended Action:** {result.recommended_action}")
        st.caption("Risk score represents the model's estimated probability of fraud.")

        st.divider()
        st.subheader("Top Risk Factors")
        exp_df = explain_row(pipeline, explainer, st.session_state["last_feature_row"])
        fig = px.bar(
            exp_df, x="shap_value", y="feature", orientation="h", color="direction",
            color_discrete_map={"increases risk": "#d73027", "decreases risk": "#1a9850"},
        )
        fig.update_layout(yaxis_title="", xaxis_title="SHAP contribution to fraud probability")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("See the 'Why is this transaction risky?' page for a full breakdown.")


# ============================================================================
# PAGE: EXPLAINABILITY
# ============================================================================
elif page == "Why is this transaction risky?":
    st.title("Why is this transaction risky?")

    if "last_result" not in st.session_state:
        st.info("Analyze a transaction on the **Transaction Analyzer** page first.")
    else:
        result = st.session_state["last_result"]
        feature_row = st.session_state["last_feature_row"]
        exp_df = explain_row(pipeline, explainer, feature_row, top_n=10)

        st.metric("Predicted Fraud Probability", f"{result.fraud_probability:.1%}")
        st.write(
            "The chart below shows the **SHAP (SHapley Additive exPlanations)** value "
            "of each feature for this specific transaction — computed directly from the "
            "trained model, not a canned explanation."
        )

        display_df = feature_row.T.reset_index()
        display_df.columns = ["feature", "value"]
        merged = exp_df.merge(display_df, on="feature", how="left")
        merged["direction"] = merged["direction"].str.replace("increases", "⬆ increases").str.replace("decreases", "⬇ decreases")

        st.dataframe(
            merged.rename(columns={
                "feature": "Feature", "value": "Value",
                "shap_value": "Contribution (SHAP)", "direction": "Effect on Risk",
            }),
            use_container_width=True, hide_index=True,
        )

        fig = go.Figure(go.Bar(
            x=merged["shap_value"], y=merged["feature"], orientation="h",
            marker_color=["#d73027" if v >= 0 else "#1a9850" for v in merged["shap_value"]],
        ))
        fig.update_layout(title="Feature contributions to fraud probability",
                           xaxis_title="SHAP value (impact on model output)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Positive SHAP values push the prediction toward fraud; negative values push "
            "it toward legitimate. Method: SHAP TreeExplainer on the trained "
            f"{summary['best_model'].replace('_', ' ')} model."
        )


# ============================================================================
# PAGE: BATCH RISK ANALYSIS
# ============================================================================
elif page == "Batch Risk Analysis":
    st.title("Batch Risk Analysis")
    st.write(
        "Upload a CSV of transactions to score them all at once. "
        f"Required columns: `{'`, `'.join(ALL_MODEL_FEATURES[:-1])}`, `type` "
        f"(must be one of {FRAUD_ELIGIBLE_TYPES})."
    )

    template = pd.DataFrame([{
        "step": 1, "type": "TRANSFER", "amount": 5000.0,
        "oldbalanceOrg": 10000.0, "newbalanceOrig": 5000.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 5000.0,
    }])
    st.download_button("Download CSV template", template.to_csv(index=False),
                        file_name="riskguard_batch_template.csv")

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read the uploaded file as CSV: {e}")
            st.stop()

        required_raw_cols = ["step", "type", "amount", "oldbalanceOrg",
                              "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
        missing = [c for c in required_raw_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Uploaded CSV is missing required columns: {missing}")
        elif batch_df.empty:
            st.error("Uploaded CSV has no rows.")
        else:
            invalid_type_mask = ~batch_df["type"].isin(FRAUD_ELIGIBLE_TYPES)
            if invalid_type_mask.any():
                st.warning(
                    f"{invalid_type_mask.sum()} row(s) have a transaction type outside "
                    f"{FRAUD_ELIGIBLE_TYPES} and will be skipped (the model only covers "
                    "these types, since fraud never occurs in the others in this dataset)."
                )
            work_df = batch_df.loc[~invalid_type_mask].copy()

            try:
                work_df["errorBalanceOrig"] = work_df["oldbalanceOrg"] - work_df["amount"] - work_df["newbalanceOrig"]
                work_df["errorBalanceDest"] = work_df["oldbalanceDest"] + work_df["amount"] - work_df["newbalanceDest"]
                scored = score_batch(pipeline, work_df[ALL_MODEL_FEATURES],
                                      st.session_state.low_threshold, st.session_state.high_threshold)
                if "nameOrig" in batch_df.columns:
                    scored.insert(0, "transaction_id", batch_df.loc[~invalid_type_mask, "nameOrig"].values)
                else:
                    scored.insert(0, "transaction_id", work_df.index)
            except Exception as e:
                st.error(f"Error while scoring batch: {e}")
                st.stop()

            st.success(f"Scored {len(scored):,} transactions.")

            filter_choice = st.selectbox(
                "Filter results", ["All", "Low Risk", "Medium Risk", "High Risk", "Predicted Fraud"]
            )
            display_df = scored.copy()
            if filter_choice == "Low Risk":
                display_df = display_df[display_df["risk_level"] == "LOW"]
            elif filter_choice == "Medium Risk":
                display_df = display_df[display_df["risk_level"] == "MEDIUM"]
            elif filter_choice == "High Risk":
                display_df = display_df[display_df["risk_level"] == "HIGH"]
            elif filter_choice == "Predicted Fraud":
                display_df = display_df[display_df["fraud_prediction"] == "POTENTIALLY FRAUDULENT"]

            show_cols = ["transaction_id", "amount", "type", "risk_score",
                         "risk_level", "fraud_prediction", "recommended_action"]
            st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)

            st.session_state["last_batch_scored"] = scored
            st.download_button(
                "Download Risk Report",
                scored.to_csv(index=False),
                file_name="riskguard_batch_results.csv",
                use_container_width=True,
            )


# ============================================================================
# PAGE: RISK REVIEW QUEUE
# ============================================================================
elif page == "Risk Review Queue":
    st.title("Risk Review Queue")
    st.caption(
        "Prototype human-in-the-loop queue. This does not connect to any real "
        "bank or Razorpay system — it simply surfaces high-risk transactions "
        "from the most recent batch analysis for manual review."
    )

    if "last_batch_scored" not in st.session_state:
        st.info("Run a **Batch Risk Analysis** first — high-risk transactions will appear here.")
    else:
        scored = st.session_state["last_batch_scored"]
        queue = scored[scored["risk_level"] == "HIGH"].sort_values("risk_score", ascending=False)

        if queue.empty:
            st.success("No high-risk transactions in the last batch. Queue is empty.")
        else:
            st.write(f"**{len(queue)}** transaction(s) awaiting review, sorted by risk score.")
            for _, row in queue.iterrows():
                with st.expander(f"Transaction {row['transaction_id']} — risk score {row['risk_score']:.1%}"):
                    cc1, cc2 = st.columns(2)
                    cc1.write(f"**Type:** {row['type']}")
                    cc1.write(f"**Amount:** {row['amount']:,.2f}")
                    cc1.write(f"**Risk Level:** {row['risk_level']}")
                    cc2.write(f"**Fraud Prediction:** {row['fraud_prediction']}")
                    cc2.write(f"**Recommended Action:** {row['recommended_action']}")

                    feature_row = pd.DataFrame([row[ALL_MODEL_FEATURES]])
                    try:
                        exp_df = explain_row(pipeline, explainer, feature_row, top_n=5)
                        st.write("**Top contributing factors:**")
                        st.dataframe(exp_df, hide_index=True, use_container_width=True)
                    except Exception:
                        st.caption("Explanation unavailable for this row.")


# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.write(
        f"Final model: **{summary['best_model'].replace('_', ' ').title()}**, "
        "selected by PR-AUC on the fraud class across three candidates "
        "(see comparison below). All metrics below are computed on a held-out "
        "test set (20% of the fraud-eligible data) that was **never used during training**."
    )

    metrics = get_best_model_metrics(summary)
    fraud_class_note = "Fraud = class 1 (isFraud), Legitimate = class 0."
    st.caption(fraud_class_note)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision (fraud)", f"{metrics['precision']:.4f}")
    c3.metric("Recall (fraud)", f"{metrics['recall']:.4f}")
    c4.metric("F1-score (fraud)", f"{metrics['f1']:.4f}")

    c5, c6 = st.columns(2)
    c5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    c6.metric("PR-AUC (Average Precision)", f"{metrics['pr_auc']:.4f}")

    st.info(
        "In fraud detection, **recall** (share of actual fraud caught) matters because missed "
        "fraud is direct financial loss, while **precision** (share of fraud alerts that are "
        "really fraud) matters because false alarms create customer friction and manual-review "
        "cost. Accuracy alone is misleading here because fraud is only "
        f"{summary['modeling_fraud_rate']:.2%} of eligible transactions — a model that predicts "
        "'never fraud' would already be >99.7% accurate while catching zero fraud."
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = np.array(metrics["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                         x=["Predicted Legitimate", "Predicted Fraud"],
                         y=["Actual Legitimate", "Actual Fraud"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("ROC Curve")
        roc = metrics["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC curve"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Precision-Recall Curve")
    pr = metrics["pr_curve"]
    fig = go.Figure(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines"))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Model Comparison")
    comp = pd.DataFrame(get_model_comparison_table(summary))
    st.dataframe(comp.style.highlight_max(subset=["precision", "recall", "f1", "roc_auc", "pr_auc"], color="#d4f4dd"),
                 use_container_width=True, hide_index=True)
    st.caption(
        f"Training set was undersampled to a 1:{summary['undersample_ratio']} fraud:legit ratio "
        f"for speed ({summary['n_train_balanced']:,} rows) — the test set of "
        f"{metrics['n_test']:,} rows ({metrics['n_test_fraud']:,} fraud) kept the real, "
        "untouched class distribution, so these metrics reflect real-world performance."
    )


# ============================================================================
# PAGE: DATA & MODEL INSIGHTS
# ============================================================================
elif page == "Data & Model Insights":
    st.title("Data & Model Insights")

    c1, c2, c3 = st.columns(3)
    c1.metric("Raw Dataset Size", f"{profile['n_rows']:,} rows")
    c2.metric("Modeling Dataset Size", f"{summary['modeling_rows']:,} rows")
    c3.metric("Number of Model Features", f"{len(ALL_MODEL_FEATURES)}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Missing Values (raw)", f"{profile['missing_values']:,}")
    c5.metric("Duplicate Rows (raw)", f"{profile['duplicate_rows']:,}")
    c6.metric("Fraud Rate (modeling set)", f"{summary['modeling_fraud_rate']:.3%}")

    st.divider()
    st.subheader("Class Imbalance Handling")
    st.write(
        f"- Train/test split: {summary['n_train']:,} / {summary['n_test']:,} rows "
        "(80/20, stratified by fraud label)\n"
        f"- Training-side undersampling of the legitimate class to a "
        f"1:{summary['undersample_ratio']} fraud:legit ratio ({summary['n_train_balanced']:,} training rows) "
        "for computational efficiency\n"
        "- Test set kept the **real, untouched** class distribution — never rebalanced\n"
        f"- Random seed fixed at {summary['random_seed']} throughout for reproducibility"
    )

    st.subheader("Model Used")
    st.write(f"**{summary['best_model'].replace('_', ' ').title()}**, selected by PR-AUC on the fraud class.")

    if summary.get("global_feature_importance"):
        st.subheader("Global Feature Importance")
        st.caption(summary.get("explainability_method", ""))
        imp_df = pd.DataFrame(summary["global_feature_importance"], columns=["feature", "mean_abs_shap"])
        imp_df["feature"] = imp_df["feature"].str.replace("num__", "").str.replace("cat__", "")
        fig = px.bar(imp_df, x="mean_abs_shap", y="feature", orientation="h")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Excluded Columns & Why")
    st.markdown(
        "- **`nameOrig`, `nameDest`** — near-unique identifier strings (millions of distinct "
        "values); would not generalize and risk ID-memorization instead of pattern learning.\n"
        "- **`isFlaggedFraud`** — derived from a simple hard-coded business rule and only ever "
        "1 for 16 of 6.36M rows; using it as a feature would leak rule-based label information.\n"
        "- **PAYMENT, CASH_IN, DEBIT transaction types** — 0 fraud cases observed across "
        "3.5M+ such transactions in this dataset; excluded from modeling (kept in the "
        "dashboard's overall statistics)."
    )


# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif page == "About":
    st.title("About RiskGuard AI")
    st.subheader('"Detect risk. Explain decisions. Protect transactions."')

    st.markdown(
        """
**What is RiskGuard AI?**
RiskGuard AI is a student-built prototype that uses machine learning to assess payment
transaction risk and identify potentially fraudulent activity. It was built for the
Razorpay AI Builder Internship 2026, Track 2: AI Risk Manager.

**Problem**
Online payment platforms process enormous transaction volumes where a small fraction
are fraudulent. Manually reviewing every transaction doesn't scale, but rule-based
systems miss evolving fraud patterns and create excessive false positives.

**Solution**
RiskGuard AI scores each transaction's fraud probability using a machine-learning model
trained on historical transaction data, converts that probability into a risk level,
explains the key drivers behind the score using SHAP, and recommends an action —
allow, verify, or hold for review.

**ML approach**
Logistic Regression, Random Forest, and HistGradientBoosting were trained and compared
on the real dataset; the model with the best PR-AUC on the minority fraud class was
selected, since accuracy is a misleading metric for this ~0.3% fraud rate.

**Explainability**
Every prediction is explained using SHAP (SHapley Additive exPlanations), computed
live against the trained model — not canned or hard-coded text.

**Risk scoring**
The model's fraud probability is mapped to LOW / MEDIUM / HIGH risk using two
configurable thresholds, documented on the Transaction Analyzer page.

**Human review**
High-risk transactions are surfaced in a Risk Review Queue as a human-in-the-loop
safety net rather than fully automated blocking.

**Limitations**
- Trained on a single simulated dataset (PaySim-style mobile money data); real payment
  data has different features and fraud patterns.
- Only covers TRANSFER and CASH_OUT transaction types, since the dataset contains no
  fraud examples in other types.
- Thresholds are illustrative, not calibrated against a real cost-of-fraud model.
- This prototype does not connect to Razorpay's or any bank's live systems.

**Disclaimer**
This is an educational prototype built for a student competition. It is **not** an
official Razorpay product, and its outputs are not real financial, banking, or
regulatory decisions.
        """
    )
