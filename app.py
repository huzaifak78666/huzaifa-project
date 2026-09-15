"""
app.py
-------
Streamlit front-end for the Fake News Detection project.
"""

import re
import string
import joblib
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection Using Machine Learning")
st.write(
    "Paste a news article's headline and/or body text below, and the model "
    "will predict whether it is likely **Real** or **Fake**."
)

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run `python train_model.py` first "
        "to train and save the model, then restart this app."
    )
    st.stop()

if "news_input" not in st.session_state:
    st.session_state.news_input = ""


def clear_text():
    st.session_state.news_input = ""


user_input = st.text_area(
    "Enter news text here:", height=200, placeholder="Paste article title/content...", key="news_input"
)

col1, col2 = st.columns(2)
with col1:
    predict_clicked = st.button("🔍 Check News", use_container_width=True)
with col2:
    st.button("🗑️ Clear", use_container_width=True, on_click=clear_text)

if predict_clicked:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0]

        confidence = max(probability) * 100

        if prediction == 1:
            st.success(f"✅ This looks like **REAL** news (confidence: {confidence:.1f}%)")
        else:
            st.error(f"⚠️ This looks like **FAKE** news (confidence: {confidence:.1f}%)")

        with st.expander("See prediction probabilities"):
            st.write(f"Fake: {probability[0]*100:.1f}%")
            st.write(f"Real: {probability[1]*100:.1f}%")

st.markdown("---")
st.caption(
    "Model: Logistic Regression + TF-IDF | Dataset: Kaggle Fake and Real News Dataset | "
    "Built with Streamlit, scikit-learn, pandas, NLTK, joblib"
)