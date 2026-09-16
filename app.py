"""
app.py - Final English Version
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

st.markdown("""
<style>
   .stTextArea textarea {
        border-radius: 12px!important;
        border: 2px solid #4A90E2!important;
    }
   .stButton button {
        border-radius: 10px!important;
        height: 48px;
        font-weight: 600!important;
    }
    div[data-testid="column"]:nth-child(1) button {
        background-color: #667eea!important;
        color: white!important;
    }
    div[data-testid="column"]:nth-child(2) button {
        background-color: #11998e!important;
        color: white!important;
    }
    div[data-testid="column"]:nth-child(3) button {
        background-color: #ff6b6b!important;
        color: white!important;
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
    else:
        st.error(f"⚠️ This looks like FAKE news (confidence: {confidence:.1f}%)")

    with st.expander("See probabilities"):
        st.write(f"Fake: {probability[0]*100:.1f}%")
        st.write(f"Real: {probability[1]*100:.1f}%")

def web_verify_and_show(text_to_check: str):
    st.subheader("🌐 Live Web Verification")
    api_key = st.secrets.get("NEWSAPI_KEY", None)
    if not api_key:
        st.error("NewsAPI key not configured. Add NEWSAPI_KEY in Secrets.")
        return
    query = text_to_check.strip()[:100]
    with st.spinner("Searching live news..."):
        try:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": query, "apiKey": api_key, "sortBy": "relevancy", "pageSize": 6, "language": "en"},
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
    for a in articles[:5]:
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

st.title("📰 Fake News Detection Using Machine Learning")
st.info("Paste a news headline or article below. The system will provide both an AI pattern-based prediction and a live web verification check.")
st.write("")

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python train_model.py` first.")
    st.stop()

if "news_input" not in st.session_state:
    st.session_state.news_input = ""

def clear_text():
    st.session_state.news_input = ""

st.write("**Enter news text here:**")
user_input = st.text_area("", height=200, placeholder="Paste article title/content...", key="news_input", label_visibility="collapsed")

col1, col2, col3 = st.columns(3)
with col1:
    predict_clicked = st.button("🤖 AI Check", use_container_width=True)
with col2:
    web_clicked = st.button("🌐 Web Verify", use_container_width=True)
with col3:
    st.button("🗑️ Clear", use_container_width=True, on_click=clear_text)

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
st.caption("Model: Logistic Regression + TF-IDF | Dataset: Kaggle Fake and Real News Dataset | Built with Streamlit, scikit-learn, NLTK, NewsAPI")