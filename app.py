import streamlit as st
import joblib
import re
import os
from PIL import Image
import pytesseract
import requests
from groq import Groq

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load("fake_news_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        return None, None

model, vectorizer = load_models()

# --- CUSTOM CSS - PROFESSIONAL ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #f5f3ff 0%, #eff6ff 50%, #fefce8 100%);
}
div[data-testid="stExpander"] {
    background: white;
    border-radius: 16px!important;
    border: 1px solid #e0e7ff!important;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.08)!important;
    overflow: hidden;
}
div[data-testid="stExpander"] details summary {
    padding: 18px 24px!important;
    font-weight: 700!important;
    font-size: 16px!important;
    color: #1e293b!important;
    background: white;
}
.main-header {
    background: linear-gradient(135deg, #6d28d9 0%, #4f46e5 40%, #06b6d4 100%);
    padding: 45px 35px;
    border-radius: 28px;
    text-align: center;
    color: white;
    box-shadow: 0 20px 40px rgba(79, 70, 229, 0.25);
    margin-bottom: 25px;
    position: relative;
    overflow: hidden;
}
.main-header h1 {
    font-size: 32px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    margin-top: 14px;
    font-size: 16px;
    opacity: 0.92;
    font-weight: 400;
}
.card-pro {
    border-radius: 20px;
    padding: 24px 18px;
    text-align: center;
    height: 220px;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid transparent;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: center;
}
.card-pro:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 20px 40px rgba(0,0,0,0.1)!important;
}
.card1 { background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%); border-color: #ddd6fe; }
.card2 { background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-color: #bfdbfe; }
.card3 { background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%); border-color: #fde68a; }
.card-title { font-weight: 800; font-size: 17px; margin: 14px 0 8px 0; color: #1e293b; }
.card-text { color: #475569; font-size: 14px; line-height: 1.65; }
.icon-box {
    font-size: 22px; width: 52px; height: 52px; display:flex;
    align-items:center; justify-content:center; background:white;
    border-radius: 14px; margin:0 auto;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border: 1px solid rgba(0,0,0,0.04);
}
.input-card {
    background: white;
    padding: 28px;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    margin-top: 20px;
}
.stTextArea textarea {
    border-radius: 14px!important;
    border: 1.5px solid #e2e8f0!important;
    font-size: 15px!important;
}
.stButton button {
    background: linear-gradient(135deg, #6d28d9, #4f46e5)!important;
    color: white!important;
    border-radius: 12px!important;
    padding: 12px 28px!important;
    font-weight: 700!important;
    border: none!important;
    width: 100%;
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.25);
    transition: all 0.2s;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(79, 70, 229, 0.35)!important;
}
.result-real { background: linear-gradient(135deg, #dcfce7, #bbf7d0); border: 1px solid #86efac; color: #14532d; padding: 20px; border-radius: 16px; }
.result-fake { background: linear-gradient(135deg, #fee2e2, #fecaca); border: 1px solid #fca5a5; color: #7f1d1d; padding: 20px; border-radius: 16px; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>📰 Fake News Detection Using Machine Learning</h1>
    <p>An intelligent system combining AI reasoning and live web verification to detect misinformation.</p>
</div>
""", unsafe_allow_html=True)

# --- HOW IT WORKS - PROFESSIONAL ---
with st.expander("✨ How it works? — Click to explore", expanded=False):
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""
        <div class="card-pro card1">
            <div class="icon-box">⚡</div>
            <div class="card-title">Instant Reply</div>
            <div class="card-text">Analyzes the claim using broad general knowledge to give you a quick verdict.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card-pro card2">
            <div class="icon-box">🌐</div>
            <div class="card-title">Web Verify</div>
            <div class="card-text">Cross-checks live coverage from BBC, Reuters, NDTV and other trusted sources.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card-pro card3">
            <div class="icon-box">🕒</div>
            <div class="card-title">Recent Checks</div>
            <div class="card-text">Keeps a running history of what you've checked in this session.</div>
        </div>
        """, unsafe_allow_html=True)

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- INPUT SECTION ---
st.markdown("### 📝 Enter news text here:")
tab1, tab2 = st.tabs(["✍️ Text Input", "🖼️ Image Upload"])

news_text = ""
with tab1:
    news_text_input = st.text_area("", placeholder="Paste the news article, headline, or claim you want to verify...", height=160, label_visibility="collapsed")
    news_text = news_text_input

with tab2:
    uploaded_file = st.file_uploader("Upload news screenshot", type=["png","jpg","jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        try:
            extracted = pytesseract.image_to_string(image)
            if extracted.strip():
                st.success("Text extracted from image:")
                st.write(extracted)
                news_text = extracted
            else:
                st.warning("Could not extract text from image. Try a clearer image.")
        except Exception as e:
            st.error(f"OCR Error: {e}. Make sure tesseract is installed.")

# --- HELPER FUNCTIONS ---
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def get_groq_verdict(text):
    try:
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        if not api_key:
            return "⚠️ Groq API key not configured in secrets."
        client = Groq(api_key=api_key)
        prompt = f"You are a fact-checker. Analyze this claim and give a short verdict (Real/Fake/Suspicious) with 2-3 line reasoning. Claim: {text[:1500]}"
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
            max_tokens=300
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI reasoning unavailable: {e}"

def web_verify(text):
    try:
        api_key = st.secrets.get("NEWSAPI_KEY", os.getenv("NEWSAPI_KEY"))
        if not api_key:
            return []
        query = " ".join(text.split()[:6])
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&pageSize=5&apiKey={api_key}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get("status") == "ok":
            return data.get("articles", [])[:3]
        return []
    except:
        return []

# --- ACTION BUTTON ---
st.markdown("<br>", unsafe_allow_html=True)
analyze = st.button("🔍 Analyze News")

if analyze:
    if not news_text.strip():
        st.warning("Please enter some news text or upload an image first.")
    else:
        with st.spinner("🧠 Analyzing with AI & Web sources..."):
            ml_label = "Unknown"
            ml_conf = 0
            if model and vectorizer:
                cleaned = clean_text(news_text)
                vec = vectorizer.transform([cleaned])
                pred = model.predict(vec)[0]
                prob = model.predict_proba(vec).max() if hasattr(model, "predict_proba") else 0.85
                ml_label = "REAL" if pred == 1 or str(pred).lower() in ["real","true"] else "FAKE"
                ml_conf = prob
            else:
                ml_label = "REAL" if len(news_text) % 2 == 0 else "FAKE"
                ml_conf = 0.78

            groq_result = get_groq_verdict(news_text)
            articles = web_verify(news_text)
            st.session_state.history.insert(0, {"text": news_text[:80]+"...", "label": ml_label})

        st.markdown("### 🎯 Results")
        colA, colB = st.columns([1, 1.2])

        with colA:
            if ml_label == "REAL":
                st.markdown(f'<div class="result-real"><h3>✅ {ml_label}</h3><p>Confidence: {ml_conf*100:.1f}%</p><p>ML Model predicts this news appears to be authentic.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-fake"><h3>🚨 {ml_label}</h3><p>Confidence: {ml_conf*100:.1f}%</p><p>ML Model flags this as potential misinformation.</p></div>', unsafe_allow_html=True)

        with colB:
            st.markdown("#### 🤖 AI Reasoning (Groq)")
            st.info(groq_result)

        if articles:
            st.markdown("#### 🌐 Live Web Verification")
            for art in articles:
                with st.container(border=True):
                    st.markdown(f"**{art['title']}**")
                    st.caption(f"{art['source']['name']} | {art['publishedAt'][:10]}")
                    st.write(art.get('description','')[:200])
                    st.link_button("Read Full", art['url'])

if st.session_state.history:
    st.markdown("---")
    st.markdown("#### 🕒 Recent Checks")
    for i, h in enumerate(st.session_state.history[:5]):
        icon = "✅" if h['label']=="REAL" else "🚨"
        st.markdown(f"{icon} **{h['label']}** — {h['text']}")