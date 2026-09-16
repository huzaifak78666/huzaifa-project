"""
app.py - Decorated Version
Fake News Detection Using Machine Learning
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

# ---------------- CUSTOM CSS FOR DECORATION ----------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown("""
<style>
   .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-attachment: fixed;
    }
    h1 {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800!important;
        padding-bottom: 10px;
    }
   .stTextArea textarea {
        border-radius: 15px!important;
        border: 2px solid #2a5298!important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        font-size: 16px!important;
    }
   .stButton button {
        border-radius: 12px!important;
        height: 50px;
        font-weight: 600!important;
        font-size: 16px!important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        border: none!important;
    }
   .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    div[data-testid="column"]:nth-child(1) button {
        background: linear-gradient(90deg, #667eea, #764ba2)!important;
        color: white!important;
    }
    div[data-testid="column"]:nth-child(2) button {
        background: linear-gradient(90deg, #11998e, #38ef7d)!important;
        color: white!important;
    }
    div[data-testid="column"]:nth-child(3) button {
        background: linear-gradient(90deg, #fc4a1a, #f7b733)!important;
        color: white!important;
    }
   .stSuccess,.stError,.stWarning {
        border-radius: 12px!important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
   .footer {
        text-align: center;
        padding: 20px;
        margin-top: 30px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

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
    cleaned = clean_text(text_to_check)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]
    confidence = max(probability) * 100

    st.markdown("### 🤖 AI Model Prediction")
    if prediction == 1:
        st.success(f"✅ **REAL NEWS** hai ye (Confidence: {confidence:.1f}%)")
        st.balloons()
    else:
        st.error(f"⚠️ **FAKE NEWS** lag raha hai (Confidence: {confidence:.1f}%)")

    with st.expander("📊 Probability dekho"):
        col1, col2 = st.columns(2)
        col1.metric("Fake %", f"{probability[0]*100:.1f}%")