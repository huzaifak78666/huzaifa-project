"""
app.py - Final Version with Colourful Header & Solid Buttons
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

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# ---------- CSS - HEADER + SOLID BUTTONS ----------
st.markdown("""
<style>
   .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #06b6d4 100%);
        padding: 35px 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(79, 70, 229, 0.25);
        margin-bottom: 25px;
        border: none;
    }
   .main-header h1 {
        font-size: 36px!important;
        color: white!important;
        margin-bottom: 10px!important;
        font-weight: 800!important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
   .main-header p {
        color: rgba(255,255,255,0.92)!important;
        font-size: 16px!important;
        max-width: 700px;
        margin: 0 auto!important;
    }
   .stTextArea textarea {
        border-radius: 14px!important;
        border: 1.5px solid #d1d5db!important;
        background: #f9fafb!important;
        font-size: 15px!important;
    }
    /* SOLID BUTTON FIX */
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
        background-color: #4f46e5!important;
        color: white!important;
        border: none!important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
        background-color: #059669!important;
        color: white!important;
        border: none!important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) button {
        background-color: #e5e7eb!important;
        color: #1f2937!important;
        border: none!important;
    }
   .stButton button {
        border-radius: 12px!important;
        height: 52px;
        font-weight: 700!important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
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

    st.subheader("🤖 AI Model Prediction")
    if prediction == 1:
        st.success(f"✅ This looks like REAL news (confidence: {confidence:.1f}%)")
        st.balloons()
        st.progress(probability[1])
    else:
        st.error(f"⚠️ This looks like FAKE news (confidence: {confidence:.1f}%)")
        st.progress(probability[0])

    with st.expander("View Detailed Analysis"):
        c1, c2 = st.columns(2)
        c1.metric("Fake Score", f"{probability[0]*100:.1f}%")
        c2.metric("Real Score", f"{probability[1]*100:.1f}%")

def web_verify_and_show(text_to_check: str):
    st.subheader("🌐 Live Web Verification")
    api_key = st.secrets.get("NEWSAPI_KEY", None)
    if not api_key:
        st.error("NewsAPI key not configured. Add NEWSAPI_KEY in Secrets.")
        return
    query = text_to_check.strip()[:120]
    with st.spinner("Checking trusted sources..."):
        try:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": query, "apiKey": api_key, "sortBy": "relevancy", "pageSize": 5, "language": "en"},
                timeout=10,
            )
            data = response.json()
        except Exception as e:
            st.warning(f"Could not search right now ({e})")
            return
    if data.get("status")!= "ok":
        st.warning(f"Error: {data.get('message', 'Unknown')}")
        return
    articles = data.get("articles", [])
    if not articles:
        st.warning("⚠️ No matching articles found online.")
        return
    trusted_hits = [a for a in articles if any(d in (a.get("url") or "") for d in TRUSTED_DOMAINS)]
    if trusted_hits:
        st.success(f"✅ Found {len(trusted_hits)} result(s) from trusted sources — likely REAL.")
    else:
        st.warning("⚠️ Found results but none from trusted sources. Verify carefully.")
    st.write("**Top results:**")
    for a in articles[:4]:
        title = a.get("title", "No title")
        url = a.get("url", "")
        source = (a.get("source") or {}).get("name", "")
        desc = a.get("description", "") or ""
        st.markdown(f"- [{title}]({url}) — *{source}*")
        if desc:
            st.caption(desc[:150] + "...")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

# ---------- HEADER ----------
st.markdown("""
<div class="main-header">
    <h1>📰 Fake News Detection Using Machine Learning</h1>
    <p>An intelligent system that combines AI pattern analysis with live web verification to detect misinformation.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ How it works?"):
    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="feature-box">🤖<br><b>AI Check</b><br>Logistic Regression + TF-IDF</div>', unsafe_allow_html=True)
    col2.markdown('<div class="feature-box">🌐<br><b>Web Verify</b><br>Checks BBC, Reuters, NDTV etc.</div>', unsafe_allow_html=True)
    col3.markdown('<div class="feature-box">📊<br><b>Confidence Score</b><br>Probability based result</div>', unsafe_allow_html=True)

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python train_model.py` first.")
    st.stop()

if "news_input" not in st.session_state:
    st.session_state.news_input = ""

def clear_text():
    st.session_state.news_input = ""

st.markdown("**Enter news text here:**")
user_input = st.text_area("", height=200, placeholder="Paste article title or full content here...", key="news_input", label_visibility="collapsed")

col1, col2, col3 = st.columns(3)
with col1:
    b1 = st.button("✨ AI Check", use_container_width=True)
with col2:
    b2 = st.button("🌐 Web Verify", use_container_width=True)
with col3:
    st.button("🗑️ Clear", use_container_width=True, on_click=clear_text)

if b1:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        predict_and_show(user_input)

if b2:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        web_verify_and_show(user_input)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#9ca3af; font-size:13px;'>Model: Logistic Regression + TF-IDF | Dataset: Kaggle | Built with Streamlit, scikit-learn & NewsAPI</p>", unsafe_allow_html=True)