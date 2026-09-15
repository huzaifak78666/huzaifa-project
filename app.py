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


def predict_and_show(text_to_check: str):
    """Runs the model on given text and displays result."""
    cleaned = clean_text(text_to_check)
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
        predict_and_show(user_input)

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
        if st.button("🔍 Check This Extracted News", use_container_width=True):
            predict_and_show(st.session_state.extracted_text)
    else:
        st.warning("Could not read any text from this image. Try a clearer, well-lit photo.")

st.markdown("---")
st.caption(
    "Model: Logistic Regression + TF-IDF | Dataset: Kaggle Fake and Real News Dataset | "
    "Built with Streamlit, scikit-learn, pandas, NLTK, joblib, Pillow, pytesseract"
)