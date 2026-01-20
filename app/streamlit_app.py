import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
import streamlit as st

from src.features import extract_features

st.set_page_config(page_title="Phishing URL Detector", layout="centered")

st.title("🔒 Phishing URL Detector")
st.write(
    "Paste a URL to predict whether it looks **phishing** or **legitimate** using engineered URL features "
    "(length, subdomains, keywords, digits, etc.)."
)

@st.cache_resource
def load_model():
    model = joblib.load("models/phishing_model.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    return model, feature_names

def risk_label(score: float) -> str:
    # score in [0,1]
    if score < 0.25:
        return "Low"
    if score < 0.50:
        return "Medium"
    if score < 0.75:
        return "High"
    return "Critical"

model, feature_names = load_model()

# --- UI layout ---
url = st.text_input("Enter URL", placeholder="https://example.com/login")

col1, col2 = st.columns([1, 2])
with col1:
    predict_btn = st.button("Predict")
with col2:
    st.caption("Tip: try `https://www.google.com` and `http://secure-login.verify-account.com/login`")

if predict_btn:
    if not url.strip():
        st.warning("Please enter a URL first.")
        st.stop()

    feats = extract_features(url)
    X = pd.DataFrame([feats])[feature_names]

    phishing_proba = float(model.predict_proba(X)[0][1])  # probability phishing

    st.subheader("Result")
    st.progress(phishing_proba)

    level = risk_label(phishing_proba)
    st.write(f"**Phishing probability:** `{phishing_proba:.2f}`  |  **Risk level:** `{level}`")

    if phishing_proba >= 0.5:
        st.error("⚠️ Likely PHISHING")
    else:
        st.success("✅ Likely LEGIT")

    st.divider()
    st.subheader("Explanation")

    # Explain via contributions (LogReg coefficients)
    clf = model.named_steps["clf"]
    coef = clf.coef_[0]
    contrib = (X.iloc[0].values * coef)

    explain_df = pd.DataFrame({
        "feature": feature_names,
        "value": X.iloc[0].values,
        "contribution": contrib
    }).sort_values("contribution", ascending=False)

    st.write("Top factors increasing phishing score:")
    st.dataframe(explain_df.head(8), use_container_width=True)

    st.write("Top factors decreasing phishing score:")
    st.dataframe(explain_df.tail(8).sort_values("contribution"), use_container_width=True)

    st.divider()
    with st.expander("View extracted features"):
        st.json(feats)
