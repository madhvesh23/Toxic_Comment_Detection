# ✅ app.py
import streamlit as st
from utils import predict_and_explain
import shap
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(page_title="Toxic Comment Detector", layout="wide")
st.title("Toxic Comment Detector with BERT + SHAP")
st.markdown("""
This app uses a fine-tuned BERT model to detect toxic comments.
It also provides interpretability using SHAP explanations and flags ambiguous or adversarial cases.
""")

# User input
user_input = st.text_area("Enter a comment:", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a comment to analyze.")
    else:
        try:
            result = predict_and_explain(user_input)

            # Prediction + Confidence
            st.markdown(f"### Prediction: **{result['prediction']}**")
            st.markdown(f"**Confidence:** {result['confidence']}%")

            # Flags
            st.markdown("### Flags")
            st.write(f"- Ambiguous Words Detected: {'Yes' if result['ambiguous_flag'] else 'No'}")
            st.write(f"- Filter Logic Flag: **{result['filter_flag']}**")

            # Top Tokens
            st.markdown("### Top Contributing Tokens")
            st.table(result['top_contributors'])

            # SHAP Visualization
            st.markdown("### SHAP Explanation")
            shap.plots.text(result['shap'], display=False)
            st.pyplot(plt.gcf())
            plt.clf()

        except Exception as e:
            st.error(f"Error: {e}")