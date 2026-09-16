"""
app.py - Final Pro Design - English
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
]

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# ---------- PRO CSS ----------
st.markdown("""
<style>
   .main-header {
        background: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.07);
        border: 1px solid #eef2f7;
        margin-bottom: 20px;
    }
   .main-header h1 {
        font-size: 36px!important;
        color: #111827!important;
        margin-bottom: 5px!important;
    }
   .main-header p {
        color: #6b7280!important;
        font-size: 15px;
    }
   .stTextArea textarea {
        border-radius: 14px!important;
        border: 1.5px solid #d1d5db!important;
        background: #f9fafb!important;
        font-size: 15px!important;
    }
   .stTextArea textarea:focus {
        border: 1.5px solid #6366f1!important;
        background: white!important;
    }
    /* Buttons */
   .stButton button {
        border-radius: 12px!important;
        height: 52px;
        font-weight: 700!important;
        letter-spacing: 0.3px;
        transition: 0.2s;
    }
    div[data-testid="column"]:nth-child(1) button {
        background: #4f46e5!important;
        color: white!important;
        border: none!important;
    }
    div[data-testid="column"]:nth-child(2) button {
        background: #059669!important;
        color: white!important;
        border: none!important;
    }
    div[data-testid="column"]:nth-child(3) button {
        background: #f3f4f6!important;
        color: #374151!important;
        border: 1px solid #e5e7eb!important;
    }
   .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    }
   .feature-box {
        background: #f8fafc;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        text-align: center;
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

    if prediction == 1:
        st.success(f"✅ **REAL NEWS** — Confidence: {confidence:.1f}%")
        st.balloons()
        st.progress(probability[1])
    else:
        st.error(f"🚨 **FAKE NEWS DETECTED** — Confidence: {confidence:.1f}%")
        st.progress(probability[0])

    with st.expander("View Detailed Analysis"):
        c1, c2 = st.columns(2)
        c1.metric("Fake Score", f"{probability[0]*100:.1f}%")
        c2.metric("Real Score", f"{probability[1]*100:.1f}%")

def web_verify_and_show(text_to_check: str):
    st.subheader("🌐 Live Web Verification")
    api_key = st.secrets.get("NEWSAPI_KEY", None)
    if not api_key:
        st.error("NewsAPI key not configured.")
        return
    query = text_to_check.strip()[:120]
    with st.spinner("Checking trusted sources..."):
        try:
            r = requests.get("https://newsapi.org/v2/everything",
                             params={"q": query, "apiKey": api_key, "sortBy": "relevancy", "pageSize": 5, "language": "en"}, timeout=10)
            data = r.json()
        except Exception as e:
            st.warning(f"Search failed: {e}")
            return
    articles = data.get("articles", [])
    if not articles:
        st.warning("⚠️ No matching articles found. This may be unverified or fabricated.")
        return
    trusted = [a for a in articles if any(d in (a.get("url") or "") for d in TRUSTED_DOMAINS)]
    if trusted:
        st.success(f"✅ Verified by {len(trusted)} trusted sources.")
    else:
        st.warning("⚠️ No trusted sources found covering this story.")
    for a in articles[:4]:
        st.markdown(f"**[{a.get('title')}]({a.get('url')})** — *{a.get('source',{}).get('name','')}*")
        st.caption((a.get('description','') or '')[:140])

@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

# ---------- HEADER ----------
st.markdown("""
<div class="main-header">
    <h1>📰 Fake News Detection Using Machine Learning</h1>
    <p>An intelligent system that combines AI pattern analysis with live web verification to detect misinformation.</p>
</div>
""", unsafe_allow_html=True)

# How it works
with st.expander("ℹ️ How it works?"):
    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="feature-box">🤖<br><b>AI Check</b><br>Logistic Regression + TF-IDF</div>', unsafe_allow_html=True)
    col2.markdown('<div class="feature-box">🌐<br><b>Web Verify</b><br>Checks BBC, Reuters, NDTV etc.</div>', unsafe_allow_html=True)
    col3.markdown('<div class="feature-box">📊<br><b>Confidence Score</b><br>Probability based result</div>', unsafe_allow_html=True)

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found.")
    st.stop()

if "news_input" not in st.session_state:
    st.session_state.news_input = ""

def clear_text():
    st.session_state.news_input = ""

st.markdown("**Enter news text here:**")
user_input = st.text_area("", height=200, placeholder="Paste article title or full content here...", key="news_input", label_visibility="collapsed")

col1, col2, col3 = st.columns(3)
with col1:
    b1 = st.button("🤖 AI Check", use_container_width=True)
with col2:
    b2 = st.button("🌐 Web Verify", use_container_width=True)
with col3:
    st.button("🗑️ Clear", use_container_width=True, on_click=clear_text)

if b1:
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        predict_and_show(user_input)
if b2:
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        web_verify_and_show(user_input)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#9ca3af; font-size:13px;'>Model: Logistic Regression + TF-IDF | Dataset: Kaggle | Built with Streamlit, scikit-learn & NewsAPI</p>", unsafe_allow_html=True)