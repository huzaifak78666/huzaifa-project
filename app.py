"""
app.py
-------
Streamlit front-end for the Fake News Detection project, styled with a
gradient banner, combining AI reasoning + live web verification + history.
"""

import re
import string
import json
import joblib
import requests
import datetime
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
    .stApp { background: #f5f6ff; }
    .banner {
        background: linear-gradient(135deg, #6d28d9 0%, #7c3aed 30%, #2563eb 70%, #06b6d4 100%);
        border-radius: 24px; padding: 35px 25px; text-align: center; color: white;
        box-shadow: 0 10px 30px rgba(109,40,217,0.3); margin-bottom: 20px;
    }
    .banner h1 { margin: 0; font-size: 27px; font-weight: 800; color: white; }
    .banner p { margin: 12px 0 0 0; opacity: 0.9; font-size: 15px; line-height: 1.5; }
    .stTextArea textarea {
        border: 2px solid #d1d5db !important; border-radius: 16px !important;
        padding: 16px !important; font-size: 15px !important; background: white !important;
    }
    div[data-testid="stButton"] button {
        border-radius: 12px !important; font-weight: 700 !important; padding: 10px !important;
        border: none !important; font-size: 14px !important;
    }
    .mk-ml + div[data-testid="stButton"] button {
        background: #4f46e5 !important; color: white !important;
    }
    .mk-web + div[data-testid="stButton"] button {
        background: #059669 !important; color: white !important;
    }
    .mk-clear + div[data-testid="stButton"] button {
        background: #e5e7eb !important; color: #374151 !important;
    }
    div[data-testid="stAlert"] { border-radius: 14px !important; }
    .info-card {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 16px;
        padding: 16px; text-align: center; margin: 6px;
    }
    .footer-caption { text-align: center; color: #6b7280; font-size: 13px; margin-top: 20px; }
    .history-item{border-radius:12px;padding:12px 16px;margin-bottom:10px;}
    .history-item.real{background:#f0fdf4;border:1px solid #bbf7d0;color:#14532d;}
    .history-item.fake{background:#fef2f2;border:1px solid #fecaca;color:#7f1d1d;}
    .history-top{display:flex;justify-content:space-between;font-size:14px;margin-bottom:4px;}
    .history-time{opacity:0.7;font-weight:400;}
    .history-text{font-size:14px;opacity:0.9;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="banner">
    <h1>📰 Fake News Detection Using Machine Learning</h1>
    <p>An intelligent system combining AI reasoning and live web verification to detect misinformation.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ How it works?"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="info-card">🧠<br><b>AI Check</b><br><small>Analyzes the claim using broad general knowledge — e.g. "The Eiffel Tower is in Paris" → verified true.</small></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="info-card">🌐<br><b>Web Verify</b><br><small>Cross-checks live coverage from BBC, Reuters, NDTV and other trusted sources.</small></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="info-card">🕘<br><b>Recent Checks</b><br><small>Keeps a running history of what you\'ve checked in this session.</small></div>', unsafe_allow_html=True)


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


def add_to_history(text, status, reason):
    if "history" not in st.session_state:
        st.session_state.history = []
    entry = {
        "text": text[:80] + ("..." if len(text) > 80 else ""),
        "status": status,
        "reason": reason,
        "time": datetime.datetime.now().strftime("%I:%M %p"),
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:8]


def render_history():
    if "history" not in st.session_state or not st.session_state.history:
        return
    st.markdown("#### 🕘 Recent Checks")
    for h in st.session_state.history:
        cls = "real" if h["status"] == "REAL" else "fake"
        icon = "✅" if h["status"] == "REAL" else "🚨"
        st.markdown(f"""
        <div class="history-item {cls}">
            <div class="history-top"><span>{icon} <b>{h['status']}</b></span><span class="history-time">{h['time']}</span></div>
            <div class="history-text">{h['text']}</div>
        </div>""", unsafe_allow_html=True)


def groq_check_and_show(text_to_check: str):
    st.markdown("#### 🧠 Result")
    groq_key = st.secrets.get("GROQ_API_KEY", None)
    if not groq_key:
        st.error("AI service not configured.")
        return
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        prompt = f"""You are a fact-checking assistant. Analyze the following statement and decide if it is REAL (factually true) or FAKE (factually false / misinformation).

Statement: "{text_to_check}"

Respond ONLY in this exact JSON format, nothing else:
{{"status": "REAL or FAKE", "reason": "A short 1-2 sentence explanation based on actual facts."}}
"""
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        status = data.get("status", "FAKE").upper()
        reason = data.get("reason", "Could not verify.")
        if status not in ("REAL", "FAKE"):
            status = "FAKE"
        if status == "REAL":
            st.success(f"✅ REAL NEWS")
        else:
            st.error(f"🚨 FAKE NEWS")
        st.markdown(f"**Summary:** {reason}")
        add_to_history(text_to_check, status, reason)
    except Exception as e:
        st.warning(f"Could not complete the AI check right now ({e}). Try again.")


def web_verify_and_show(text_to_check: str):
    st.markdown("#### 🌐 Live Web Verification")
    api_key = st.secrets.get("NEWSAPI_KEY", None)
    if not api_key:
        st.error("Web verification service not configured.")
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
            st.warning(f"Web search failed ({e}). Try again.")
            return
    if data.get("status") != "ok":
        st.warning(f"Search error: {data.get('message', 'Unknown error')}")
        return
    articles = [a for a in data.get("articles", []) if a.get("description")]
    if not articles:
        st.warning("⚠️ No matching coverage found online.")
        return
    trusted_hits = [a for a in articles if any(d in (a.get("url") or "") for d in TRUSTED_DOMAINS)]
    if trusted_hits:
        st.success(f"✅ Found {len(trusted_hits)} result(s) from trusted sources.")
    else:
        st.warning("⚠️ No trusted sources found matching this story.")
    for a in articles[:4]:
        source = (a.get("source") or {}).get("name", "")
        st.markdown(f"- [{a.get('title','No title')}]({a.get('url','')}) — *{source}*")


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


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
user_input = st.text_area("news_input_label", height=200, placeholder="Paste article title/content...",
                           key="news_input", label_visibility="collapsed")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="mk-ml"></div>', unsafe_allow_html=True)
    check_clicked = st.button("✨ Check News", use_container_width=True)
with col2:
    st.markdown('<div class="mk-web"></div>', unsafe_allow_html=True)
    web_clicked = st.button("🌐 Web Verify", use_container_width=True)
with col3:
    st.markdown('<div class="mk-clear"></div>', unsafe_allow_html=True)
    st.button("🗑️ Clear", use_container_width=True, on_click=clear_text)

if check_clicked:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        groq_check_and_show(user_input)

if web_clicked:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        web_verify_and_show(user_input)

render_history()

st.markdown(
    '<p class="footer-caption">Built with Streamlit, scikit-learn, pandas, NLTK, joblib, and AI-powered reasoning</p>',
    unsafe_allow_html=True,
)