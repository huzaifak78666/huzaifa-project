"""
app.py
-------
Streamlit front-end for the Fake News Detection project.
"""

import re
import string
import joblib
import streamlit as st
from PIL import Image
import pytesseract
from duckduckgo_search import DDGS

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
    "cnn.com", "aljazeera.com", "theguardian.com", "npr.org", "pti.co.in",
    "livemint.com", "business-standard.com",
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
        st.error(f"⚠️ This looks like **FAKE** news (confidence: {confidence:.1f}%)")

    with st.expander("See prediction probabilities"):
        st.write(f"Fake: {probability[0]*100:.1f}%")
        st.write(f"Real: {probability[1]*100:.1f}%")


def web_verify_and_show(text_to_check: str):
    """Searches the live web to see if trusted news sources cover this story."""
    st.subheader("🌐 Live Web Verification")
    query = text_to_check.strip()[:150]  # keep query short

    with st.spinner("Searching the web for matching news..."):
        try:
            results = list(DDGS().text(query, max_results=8))
        except Exception as e:
            st.warning(f"Could not complete web search right now ({e}). Try again in a moment.")
            return

    if not results:
        st.warning(
            "⚠️ No matching articles found online. This could mean the news is "
            "very new, very obscure, or possibly fabricated."
        )
        return

    trusted_hits = [r for r in results if any(d in r.get("href", "") for d in TRUSTED_DOMAINS)]

    if trusted_hits:
        st.success(f"✅ Found {len(trusted_hits)} matching result(s) from trusted news sources — likely REAL.")
    else:
        st.warning(
            "⚠️ Found some results, but none from well-known trusted news sources. "
            "Verify carefully before believing this."
        )

    st.write("**Top search results:**")
    for r in results[:5]:
        title = r.get("title", "No title")
        href = r.get("href", "")
        body = r.get("body", "")
        st.markdown(f"- [{title}]({href})")
        if body:
            st.caption(body[:150] + "...")


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

# ---------------------------------------------------------------------------
# Image / Photo input section
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📷 Or check a news photo/screenshot")

img_option = st.radio("Choose input method:", ["📁 Upload from gallery", "📸 Take a photo"], horizontal=True)

image_file = None
if img_option == "📁 Upload from gallery":
    image_file = st.file_uploader("Upload an image (screenshot of a news article)", type=["png", "jpg", "jpeg"])
else:
    image_file = st.camera_input("Take a photo of the news article")

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
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            if st.button("🤖 AI Check This Text", use_container_width=True):
                predict_and_show(st.session_state.extracted_text)
        with ecol2:
            if st.button("🌐 Web Verify This Text", use_container_width=True):
                web_verify_and_show(st.session_state.extracted_text)
    else:
        st.warning("Could not read any text from this image. Try a clearer, well-lit photo.")

st.markdown("---")
st.caption(
    "Model: Logistic Regression + TF-IDF | Dataset: Kaggle Fake and Real News Dataset | "
    "Built with Streamlit, scikit-learn, pandas, NLTK, joblib, Pillow, pytesseract, duckduckgo-search"
)