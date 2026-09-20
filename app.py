"""
app.py
-------
Streamlit front-end for the Fake News Detection project.
Combines: trained ML model (TF-IDF + Logistic Regression, 99% accuracy),
Groq AI reasoning-based check, live web verification, and photo/OCR upload.
"""

import re
import string
import joblib
import requests
import streamlit as st
from PIL import Image
import pytesseract
import json

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


# ---------------------------------------------------------------------------
# CHECK 1: Trained ML Model (Kaggle dataset, TF-IDF + Logistic Regression)
# ---------------------------------------------------------------------------
def predict_and_show(text_to_check: str):
    cleaned = clean_text(text_to_check)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]
    confidence = max(probability) * 100

    st.subheader("🤖 ML Model Prediction (trained on Kaggle dataset)")
    if prediction == 1:
        st.success(f"✅ This looks like **REAL** news (confidence: {confidence:.1f}%)")
    else:
        st.error(f"⚠️ This looks like **FAKE** news (confidence: {confidence:.1f}%)")

    with st.expander("See prediction probabilities"):
        st.write(f"Fake: {probability[0]*100:.1f}%")
        st.write(f"Real: {probability[1]*100:.1f}%")

    st.caption(
        "This check recognizes writing-style patterns learned from the training dataset. "
        "It is most reliable for news similar in style/topic to the training data."
    )


# ---------------------------------------------------------------------------
# CHECK 2: Groq AI Reasoning (real-world fact-based reasoning)
# ---------------------------------------------------------------------------
def groq_check_and_show(text_to_check: str):
    st.subheader("🧠 AI Reasoning Check (Groq / Llama 3)")

    groq_key = st.secrets.get("GROQ_API_KEY", None)
    if not groq_key:
        st.error("GROQ_API_KEY not configured. Add it in the app's Secrets settings.")
        return

    try:
        from groq import Groq
        client = Groq(api_key=groq_key)

        prompt = f"""You are a fact-checking assistant. Analyze the following statement and decide if it is REAL (factually true / accurate) or FAKE (factually false / misinformation).

Statement: "{text_to_check}"

Respond ONLY in this exact JSON format, nothing else, no markdown, no extra text:
{{"status": "REAL or FAKE", "reason": "A short 1-2 sentence explanation of why, based on actual facts."}}
"""
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        status = data.get("status", "FAKE").upper()
        reason = data.get("reason", "Could not verify.")
        if status not in ("REAL", "FAKE"):
            status = "FAKE"

        if status == "REAL":
            st.success(f"✅ **REAL** — {reason}")
        else:
            st.error(f"🚨 **FAKE** — {reason}")

        st.caption("This check uses an AI language model's general world knowledge to reason about the claim, rather than pattern-matching against the training dataset.")
    except Exception as e:
        st.warning(f"Could not complete AI reasoning check right now ({e}). Please try again.")


# ---------------------------------------------------------------------------
# CHECK 3: Live Web Verification (NewsAPI)
# ---------------------------------------------------------------------------
def web_verify_and_show(text_to_check: str):
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
                    "q": query, "apiKey": api_key, "sortBy": "relevancy",
                    "pageSize": 6, "language": "en",
                },
                timeout=10,
            )
            data = response.json()
        except Exception as e:
            st.warning(f"Could not complete web search right now ({e}). Try again in a moment.")
            return

    if data.get("status") != "ok":
        st.warning(f"Search service returned an error: {data.get('message', 'Unknown error')}")
        return

    articles = [a for a in data.get("articles", []) if a.get("description")]
    if not articles:
        st.warning("⚠️ No matching coverage found online. This could mean the news is very new, obscure, or possibly fabricated.")
        return

    trusted_hits = [a for a in articles if any(d in (a.get("url") or "") for d in TRUSTED_DOMAINS)]
    if trusted_hits:
        st.success(f"✅ Found {len(trusted_hits)} matching result(s) from trusted news sources.")
    else:
        st.warning("⚠️ Found some results, but none from well-known trusted sources.")

    st.write("**Top search results:**")
    for a in articles[:4]:
        title = a.get("title", "No title")
        url = a.get("url", "")
        source = (a.get("source") or {}).get("name", "")
        st.markdown(f"- [{title}]({url}) — *{source}*")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection Using Machine Learning")
st.write(
    "Paste a news headline or article below. Choose an ML-based check (trained on the "
    "Kaggle dataset), an AI reasoning check, or a live web verification."
)

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Please run `python train_model.py` first.")
    st.stop()

if "news_input" not in st.session_state:
    st.session_state.news_input = ""


def clear_text():
    st.session_state.news_input = ""


user_input = st.text_area(
    "Enter news text here:", height=200, placeholder="Paste article title/content...", key="news_input"
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    ml_clicked = st.button("🤖 ML Check", use_container_width=True)
with col2:
    ai_clicked = st.button("🧠 AI Reasoning", use_container_width=True)
with col3:
    web_clicked = st.button("🌐 Web Verify", use_container_width=True)
with col4:
    st.button("🗑️ Clear", use_container_width=True, on_click=clear_text)

if ml_clicked:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        predict_and_show(user_input)

if ai_clicked:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        groq_check_and_show(user_input)

if web_clicked:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        web_verify_and_show(user_input)

# ---------------------------------------------------------------------------
# Photo / Screenshot Upload
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📷 Or check a news photo/screenshot")

image_file = st.file_uploader("Upload an image (screenshot of a news article)", type=["png", "jpg", "jpeg"])

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Selected Image", use_container_width=True)

    with st.spinner("Reading text from image..."):
        try:
            extracted_text = pytesseract.image_to_string(image)
        except Exception as e:
            extracted_text = ""
            st.error(f"OCR error: {e}")

    if extracted_text.strip():
        st.text_area("Extracted Text (editable):", extracted_text, height=150, key="extracted_text")
        ecol1, ecol2, ecol3 = st.columns(3)
        with ecol1:
            if st.button("🤖 ML Check This Text", use_container_width=True):
                predict_and_show(st.session_state.extracted_text)
        with ecol2:
            if st.button("🧠 AI Reasoning on This", use_container_width=True):
                groq_check_and_show(st.session_state.extracted_text)
        with ecol3:
            if st.button("🌐 Web Verify This Text", use_container_width=True):
                web_verify_and_show(st.session_state.extracted_text)
    else:
        st.warning("Could not read any text from this image. Try a clearer, well-lit photo.")

st.markdown("---")
st.caption(
    "Model: Logistic Regression + TF-IDF | Dataset: Kaggle Fake and Real News Dataset | "
    "AI Reasoning: Llama 3 via Groq | Built with Streamlit, scikit-learn, pandas, NLTK, joblib, pytesseract"
)