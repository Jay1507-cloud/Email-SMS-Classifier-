# app.py
import streamlit as st
import pickle
import nltk
import string
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.utils.validation import check_is_fitted
import os

BASE = Path(__file__).resolve().parent
ps = PorterStemmer()

st.set_page_config(page_title="Email/SMS Spam Classifier", page_icon="📩", layout="wide")

# Load external CSS
def load_css(fname="style.css"):
    css_path = BASE / fname
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>body{background:#fff}</style>", unsafe_allow_html=True)

load_css("style.css")

# NLTK (quiet)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Text transform
def transform_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered = [t for t in tokens if t.isalnum()]
    filtered = [t for t in filtered if t not in stopwords.words("english") and t not in string.punctuation]
    stems = [ps.stem(w) for w in filtered]
    return " ".join(stems)

# Load model and vectorizer (cached)
@st.cache_resource
def load_model_and_vectorizer():
    vfile = BASE / "vectorizer.pkl"
    mfile = BASE / "model.pkl"
    missing = []
    if not vfile.exists():
        missing.append(str(vfile.name))
    if not mfile.exists():
        missing.append(str(mfile.name))
    if missing:
        return None, None, f"Missing files: {', '.join(missing)}"
    try:
        vectorizer = pickle.load(open(vfile, "rb"))
        model = pickle.load(open(mfile, "rb"))
        return model, vectorizer, None
    except Exception as e:
        return None, None, f"Error loading pickles: {e}"

with st.spinner("Loading model and resources..."):
    model, vectorizer, load_err = load_model_and_vectorizer()

# Header (navbar-like)
st.markdown("""
<header class="nav">
  <div class="nav-left">
    <div class="brand">📩 Spam Classifier</div>
    <div class="nav-sub">Email / SMS detection</div>
  </div>
  <div class="nav-right">
    <div class="nav-item">Home</div>
    <div class="nav-item">About</div>
    <div class="nav-item">Contact</div>
  </div>
</header>
""", unsafe_allow_html=True)

if load_err:
    st.error(load_err)
    st.info("Place model.pkl and vectorizer.pkl into the same folder as app.py, then restart the app.")
    st.stop()

# Check fitted
try:
    check_is_fitted(model)
    fitted = True
except Exception:
    fitted = False

if not fitted:
    st.warning("Model loaded but appears not fitted. Retrain & save a fitted model.pkl.")
    st.stop()

# Layout
left, right = st.columns([3,1.15])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Enter the message")
    message = st.text_area("", placeholder="Type or paste SMS / email message here...", height=180)

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        predict = st.button("Predict")
    with c2:
        clear = st.button("Clear")
    with c3:
        demo = st.button("Demo spam")

    if demo:
        message = "Congratulations! You have won a $1000 prize! Click here to claim."

    if clear:
        st.experimental_rerun()

    result_placeholder = st.empty()

    if predict:
        if not message or message.strip() == "":
            st.warning("Please enter a message.")
        else:
            with st.spinner("Analyzing..."):
                transformed = transform_text(message)
                vec = vectorizer.transform([transformed])
                conf = None
                try:
                    probs = model.predict_proba(vec)
                    conf = float(probs[0].max())
                except Exception:
                    conf = None
                label = model.predict(vec)[0]

            if label == 1:
                result_placeholder.markdown('<div class="result-danger">🚫 This message is <b>SPAM</b></div>', unsafe_allow_html=True)
            else:
                result_placeholder.markdown('<div class="result-success">✅ This message is <b>NOT SPAM</b></div>', unsafe_allow_html=True)

            st.markdown(f"**Confidence:** {(conf*100):.1f}%" if conf is not None else "**Confidence:** N/A")
            st.markdown("**Transformed text:**")
            st.code(transformed or "(empty after preprocessing)")

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("About")
    st.write("- Model: MultinomialNB")
    st.write("- Vectorizer: TF-IDF")
    st.write("- Preprocessing: tokenization, stopwords, stemming")
    st.markdown("---")
    st.subheader("Files")
    files = [f for f in os.listdir(BASE) if f.endswith((".pkl", ".csv", ".py"))]
    st.write(", ".join(files))

st.markdown("<hr><div class='footer'>Made with ❤️ • Dhananjay</div>", unsafe_allow_html=True)
