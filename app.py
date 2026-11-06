import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "random_forest_sms_model.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Add random_forest_sms_model.pkl "
            "to the same GitHub repository as app.py."
        )
    return joblib.load(MODEL_PATH)

st.title("📩 SMS Spam Detection")
st.write(
    "Enter an SMS message below. The Random Forest model will "
    "classify it as Spam or Ham (normal message)."
)

try:
    model = load_model()
except Exception as error:
    st.error(str(error))
    st.stop()

message = st.text_area(
    "Enter your SMS message",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=140
)

if st.button("Predict Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)]) * 100

        st.subheader("Prediction Result")

        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ HAM — NORMAL MESSAGE")

        st.metric("Model confidence", f"{confidence:.2f}%")
        st.caption(
            "Confidence is the model's estimated probability, not a guarantee "
            "that a message is safe or fraudulent."
        )

with st.expander("About this project"):
    st.write(
        "This educational project uses a TF-IDF text vectorizer and a "
        "Random Forest classifier trained on the SMS Spam Collection dataset."
    )
