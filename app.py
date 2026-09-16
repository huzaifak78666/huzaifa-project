"""
app.py

Streamlit front-end for the Fake News Detection project.
"""

import re
import string
import joblib
import requests
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

TRUSTED_DOMAINS = [
    "bbc.com", "reuters.com", "apnews.com", "ndtv.com", "thehindu.com",
    "timesofindia.indiatimes.com", "indianexpress.com", "hindustantimes.com",
    "cnn.com", "aljazeera.com", "theguardian.com", "npr.org",
    "livemint.com", "business-standard.com", "news18.com",
]

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

def predict_and_show(text_to_check: str):
    """Runs the ML model on given text and displays result."""
    cleaned = clean_text(text_to_check)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]
    confidence = max(probability) * 100

    st.subheader("🤖 AI Model Prediction (pattern-based)")
    if prediction == 1:
        st.success(f"✅ This looks like **REAL** news (confidence: {confidence:.1f}%)")
    else:
        st.error(f"⚠ This looks like **FAKE** news (confidence: {confidence:.1f}%)")

    with st.expander("See prediction probabilities"):
        st.write(f"Fake: {probability[0]*100:.1f}%")
        st.write(f"Real: {probability[1]*100:.1f}%")

def web_verify_and_show(text_to_check: str):
    """Searches live news via NewsAPI to see if trusted sources cover this story."""
    st.subheader("🌐 Live Web Verification")

    api_key = st.secrets.get("NEWSAPI_KEY", None)
    if not api_key:
        st.error("NewsAPI key not configured. Add NEWSAPI_KEY in app Secrets settings.")
        return

    query = text_to_check.strip()[:100]

    with st.spinner("Searching live news for matching stories..."):
        try:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "apiKey": api_key,
                    "sortBy": "relevancy",
                    "pageSize": 6,
                    "language": "en",
                },
                timeout=10,
            )
            data = response.json()
        except Exception as e:
            st.warning(f"Could not complete web search right now ({e}). Try again in a moment.")
            return

    if data.get("status")!= "ok":
        st.warning(f"Search service returned an error: {data.get('message', 'Unknown error')}")
        return

    articles = data.get("articles", [])

    if not articles:
        st.warning(
            "⚠ No matching articles found online. This could mean the news is "
            "very new, very obscure, or possibly fabricated."
        )
        return

    trusted_hits = [
        a for a in articles
        if any(d in (a.get("url") or "") for d in TRUSTED_DOMAINS)
    ]

    if trusted_hits:
        st.success(f"✅ Found {len(trusted_hits)} matching result(s) from trusted news sources — likely REAL.")
    else:
        st.warning(
            "⚠ Found some results, but none from well-known trusted news sources. "
            "Verify carefully before believing this."
        )

    st.write("**Top search results:**")
    for a in articles[:5]:
        title = a.get("title", "No title")
        url = a.get("url", "")
        source = (a.get("source") or {}).get("name", "")
        description = a.get("description", "") or ""
        st.markdown(f"- [{title}]({url}) — *{source}*")
        if description:
            st.caption(description[:150] + "...")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection Using Machine Learning")
st.write(
    "Paste a news article's headline and/or body text below. The app will give you "
    "both an **AI pattern-based prediction** and a **live web verification check**."
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

col1, col2, col3 = st.columns(3)
with col1:
    predict_clicked = st.button("🤖 AI Check", use_container_width=True)
with col2:
    web_clicked = st.button("🌐 Web Verify", use_container_width=True)
with col3:
    st.button("🗑 Clear", use_container_width=True, on_click=clear_text)

if predict_clicked:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        predict_and_show(user_input)

if web_clicked:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        web_verify_and_show(user_input)

st.markdown("---")
st.caption(
    "Model: Logistic Regression + TF-IDF | Dataset: Kaggle Fake and Real News Dataset | "
    "Built with Streamlit, scikit-learn, pandas, NLTK, joblib, Pillow, pytesseract, NewsAPI"
)