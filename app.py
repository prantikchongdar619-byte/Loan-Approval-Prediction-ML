import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Risk Scorer",
    page_icon="🏦",
    layout="wide"
)

# ── Load model & scaler ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# ── Preprocessing (mirrors your notebook pipeline exactly) ─────────────────────
def preprocess_data(df):
    df = df.copy()

    if 'id' in df.columns:
        df.set_index('id', inplace=True)

    # Interaction features
    df['income_to_emp_length']    = df['person_income'] / df['person_emp_length'].replace(0, 1)
    df['income_to_age']           = df['person_income'] / df['person_age']
    df['loan_amt_times_int_rate'] = df['loan_amnt'] * df['loan_int_rate']

    # Decile binning
    def safe_qcut(series, q=10):
        try:
            return pd.qcut(series, q=q, labels=False, duplicates='drop') + 1
        except ValueError:
            return pd.cut(series, bins=q, labels=False, duplicates='drop') + 1

    for col, alias in [
        ('person_age',                 'age_decile'),
        ('person_income',              'income_decile'),
        ('person_emp_length',          'emp_length_decile'),
        ('loan_percent_income',        'loan_percent_income_decile'),
        ('loan_amnt',                  'loan_amnt_decile'),
        ('loan_int_rate',              'loan_int_rate_decile'),
        ('cb_person_cred_hist_length', 'cb_person_cred_hist_length_decile'),
    ]:
        df[alias] = safe_qcut(df[col])

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=['person_home_ownership', 'loan_intent', 'cb_person_default_on_file'],
        drop_first=True,
        dtype=int
    )

    # Label encode loan_grade
    le = LabelEncoder()
    df['loan_grade_encoded'] = le.fit_transform(df['loan_grade'])

    # Drop raw columns
    cols_to_drop = [
        'person_age', 'person_income', 'person_emp_length',
        'loan_grade', 'loan_amnt', 'loan_int_rate',
        'loan_percent_income', 'cb_person_cred_hist_length'
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    df.dropna(inplace=True)

    return df


def align_columns(df, reference_columns):
    """
    Ensure the uploaded CSV has the same columns the model was trained on.
    Adds missing columns as 0, drops extras.
    """
    for col in reference_columns:
        if col not in df.columns:
            df[col] = 0
    return df[reference_columns]


def assign_risk_tier(score):
    if score < 0.3:
        return "🟢 Low"
    elif score < 0.6:
        return "🟡 Medium"
    else:
        return "🔴 High"


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏦 Loan Risk Batch Scorer")
st.markdown("Upload a CSV of loan applications to get default probability scores, risk tiers, and approve/reject decisions.")

# Sidebar controls
st.sidebar.header("Scoring Settings")

threshold = st.sidebar.slider(
    "Approval Threshold (PD Score)",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Customers with PD score below this threshold are approved."
)

strategy_label = (
    "🛡️ Conservative"  if threshold <= 0.40 else
    "⚖️ Balanced"       if threshold <= 0.65 else
    "🚀 Aggressive"
)
st.sidebar.markdown(f"**Strategy:** {strategy_label}")
st.sidebar.markdown("""
| Strategy | Threshold |
|---|---|
| 🛡️ Conservative | ≤ 0.40 |
| ⚖️ Balanced | 0.41 – 0.65 |
| 🚀 Aggressive | > 0.65 |
""")

lgd = st.sidebar.slider(
    "Loss Given Default (LGD)",
    min_value=0.3,
    max_value=0.9,
    value=0.6,
    step=0.1,
    help="Fraction of loan amount lost if customer defaults."
)

# File upload
uploaded_file = st.file_uploader("Upload loan applications CSV", type=["csv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("Preview — Uploaded Data")
    st.dataframe(raw_df.head(), use_container_width=True)
    st.caption(f"{len(raw_df):,} applications loaded")

    with st.spinner("Preprocessing and scoring..."):
        # Keep original columns for output display
        display_cols = ['person_age', 'person_income', 'loan_amnt',
                        'loan_int_rate', 'loan_grade', 'loan_intent',
                        'person_home_ownership']
        display_cols = [c for c in display_cols if c in raw_df.columns]
        display_df = raw_df[display_cols].copy()

        # Preprocess
        processed = preprocess_data(raw_df)

        # Align to training columns
        train_columns = list(processed.columns)
        processed = align_columns(processed, train_columns)

        # Scale
        processed_scaled = scaler.transform(processed)

        # Score
        pd_scores = model.predict_proba(processed_scaled)[:, 1]

    # Build results dataframe
    ead = raw_df['loan_amnt']
    results_df = display_df.copy()
    results_df['PD Score']       = pd_scores.round(4)
    results_df['Risk Tier']      = [assign_risk_tier(s) for s in pd_scores]
    results_df['Expected Loss']  = (pd_scores * lgd * ead).round(2)
    results_df['Decision']       = np.where(pd_scores < threshold, '✅ Approve', '❌ Reject')

    # ── Summary metrics ────────────────────────────────────────────────────────
    st.subheader("Portfolio Summary")

    approved    = (pd_scores < threshold)
    total       = len(pd_scores)
    n_approved  = approved.sum()
    n_rejected  = total - n_approved
    total_el    = (pd_scores * lgd * ead).sum()
    approved_el = (pd_scores[approved] * lgd * ead[approved]).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applications", f"{total:,}")
    col2.metric("Approved",           f"{n_approved:,}",  f"{n_approved/total:.1%}")
    col3.metric("Rejected",           f"{n_rejected:,}",  f"{n_rejected/total:.1%}")
    col4.metric("Approved Portfolio Expected Loss", f"₹{approved_el:,.0f}")

    # Risk tier breakdown
    st.subheader("Risk Tier Breakdown")
    tier_counts = results_df['Risk Tier'].value_counts().reset_index()
    tier_counts.columns = ['Risk Tier', 'Count']
    st.dataframe(tier_counts, use_container_width=True, hide_index=True)

    # ── Results table ──────────────────────────────────────────────────────────
    st.subheader("Scored Applications")

    # Filter controls
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        decision_filter = st.selectbox(
            "Filter by Decision",
            ["All", "✅ Approve", "❌ Reject"]
        )
    with filter_col2:
        tier_filter = st.selectbox(
            "Filter by Risk Tier",
            ["All", "🟢 Low", "🟡 Medium", "🔴 High"]
        )

    filtered = results_df.copy()
    if decision_filter != "All":
        filtered = filtered[filtered['Decision'] == decision_filter]
    if tier_filter != "All":
        filtered = filtered[filtered['Risk Tier'] == tier_filter]

    st.dataframe(
        filtered.sort_values('PD Score', ascending=False),
        use_container_width=True,
        hide_index=True
    )

    # ── Download ───────────────────────────────────────────────────────────────
    st.subheader("Download Results")
    csv_out = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Scored CSV",
        data=csv_out,
        file_name="loan_scoring_results.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload a CSV to get started. Expected columns match the Kaggle Loan Approval dataset format.")

    # Show expected columns as a guide
    with st.expander("Expected CSV columns"):
        st.code("""person_age, person_income, person_emp_length, loan_amnt,
loan_int_rate, loan_percent_income, loan_grade, loan_intent,
person_home_ownership, cb_person_default_on_file, cb_person_cred_hist_length""")