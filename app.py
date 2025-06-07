import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle
import os
import base64
import gdown
import zipfile

# --- Download model if not exists ---
def download_and_extract_model():
    if not os.path.exists("model"):
        file_id = "1vQkKsrnKFwE7VNzKYfDrOaOtWNr1CL5C"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "model.zip"

        gdown.download(url, output, quiet=False)

        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove(output)

download_and_extract_model()

# --- Load Models ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("model/bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("model").to(device)
sbert_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# Load embeddings and dictionary
with open("model/real_embeddings.pkl", "rb") as f:
    real_embeddings = pickle.load(f)
with open("model/fake_embeddings.pkl", "rb") as f:
    fake_embeddings = pickle.load(f)
with open("model/exact_match_dict.pkl", "rb") as f:
    exact_match_dict = pickle.load(f)
# --- Streamlit Page Config ---
st.set_page_config(
    page_title="📰 Fake News Detector",
    layout="centered",
)

# --- Set Background ---
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("wallpaper1.jpg")

# --- Custom CSS ---
st.markdown(
    """
    <style>
        .main {
            background-color: rgba(0, 0, 0, 0.7);
            padding: 2rem;
            border-radius: 10px;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(270deg, #2afa05, #fafa05, #05f2fa, #fa05d9);
            background-size: 600% 600%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: animateGradient 8s ease infinite;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            transform-style: preserve-3d;
            transition: transform 0.2s ease;
        }
        .title:hover {
            transform: perspective(500px) rotateY(5deg) rotateX(5deg);
        }
        @keyframes animateGradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .textbox-label {
            display: inline-block;
            transform-style: preserve-3d;
            transition: transform 0.2s ease;
            color: #e8faf9;
        }
        .textbox-label:hover {
            transform: perspective(500px) rotateY(5deg) rotateX(5deg);
        }
        .stButton>button {
            background: linear-gradient(to right, #ff4b2b,#e3e075);
            color: white !important;
            border: none;
            padding: 0.75em 2em;
            font-size: 1em;
            font-weight: bold;
            border-radius: 12px;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            cursor: pointer;
        }
        div[data-baseweb="textarea"] {
            transition: all 0.3s ease-in-out;
            border-radius: 12px;
            border: 2px solid #e34f4a;
            padding: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            background-color: #121212;
        }
        div[data-baseweb="textarea"]:hover {
            transform: scale(1.01);
            box-shadow: 0 6px 20px rgba(255, 187, 108, 1);
            border-color: #ffbb6c;
            cursor: pointer;
        }
        textarea {
            color: white;
            background-color: transparent;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.markdown('<div class="title">🧠 Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="textbox-label"><h4>💬 Enter a news article below to check if it\'s Real or Fake</h4></div>', unsafe_allow_html=True)
st.markdown('<div class="textbox-label"> <h4>✍️ Type or paste the news article here (Dataset):</h4></div>', unsafe_allow_html=True)
# --- Persistent Real News Store ---
REAL_NEWS_STORE = "stored_real_news.pkl"

def load_stored_real_news():
    if os.path.exists(REAL_NEWS_STORE):
        with open(REAL_NEWS_STORE, "rb") as f:
            return pickle.load(f)
    return []

def save_stored_real_news(news_list):
    with open(REAL_NEWS_STORE, "wb") as f:
        pickle.dump(news_list, f)

stored_real_news = load_stored_real_news()

# --- Prediction with Scores ---
def predict_news(text):
    text = text.strip()

    # Exact match check
    exact_match_score = 1.0 if text in exact_match_dict else 0.0
    if exact_match_score == 1.0:
        return "✅ Real", exact_match_score, None, None, None

    input_embedding = sbert_model.encode(text, convert_to_tensor=True).to(device)

    # Semantic similarity with stored real news
    max_sim_to_real_stored = 0
    for real_news in stored_real_news:
        real_embedding = sbert_model.encode(real_news, convert_to_tensor=True).to(device)
        sim = util.cos_sim(input_embedding, real_embedding).item()
        max_sim_to_real_stored = max(max_sim_to_real_stored, sim)

    if max_sim_to_real_stored >= 0.8:
        return "❌ Fake (Modified from Real)", 0.0, max_sim_to_real_stored, None, None

    # Semantic similarity with real/fake datasets
    real_tensor = torch.tensor(real_embeddings).to(device)
    fake_tensor = torch.tensor(fake_embeddings).to(device)

    real_sim = util.cos_sim(input_embedding, real_tensor).max().item()
    fake_sim = util.cos_sim(input_embedding, fake_tensor).max().item()

    if abs(real_sim - fake_sim) > 0.05:
        is_real = real_sim > fake_sim
        if is_real:
            if text not in stored_real_news:
                stored_real_news.append(text)
                save_stored_real_news(stored_real_news)
            return "✅ Real News", 0.0, max_sim_to_real_stored, real_sim, fake_sim
        else:
            return "❌ Fake News", 0.0, max_sim_to_real_stored, real_sim, fake_sim

    # Classifier fallback
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs).item()

    if label == 1:
        if text not in stored_real_news:
            stored_real_news.append(text)
            save_stored_real_news(stored_real_news)
        return "✅ Real News", 0.0, max_sim_to_real_stored, real_sim, fake_sim
    else:
        return "❌ Fake News", 0.0, max_sim_to_real_stored, real_sim, fake_sim
# --- User Input ---
with st.container():
    user_input = st.text_area(
        "",
        height=200,
        key="textbox",
        help="Enter the news to check if it's fake or real",
        placeholder="Please enter the news article here..."
    )

# --- Predict Button ---
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.markdown(f"""
            <div style="
                margin-top: 30px;
                padding: 1.2rem;
                border-radius: 12px;
                background-color: rgba(0, 0, 0, 0.8);
                border: 3px solid #ffb02e;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
                text-align: center;
            ">
            <span style="font-size: 28px; font-weight: bold; color: #ffb02e;">
                ⚠️ Please enter some text to predict.
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("🔎 Analyzing..."):
            prediction, exact_score, sim_stored_real, real_score, fake_score = predict_news(user_input)
            is_real = prediction.startswith("✅")
            output_color = "#00cc66" if is_real else "#ff4d4d"
            output_icon = "✅" if is_real else "❌"

            st.markdown(f"""
                <div style="
                    margin-top: 30px;
                    padding: 1.2rem;
                    border-radius: 12px;
                    background-color: rgba(0, 0, 0, 0.8);
                    border: 3px solid {output_color};
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
                    text-align: center;
                ">
                <span style="font-size: 28px; font-weight: bold; color: {output_color};">
                    {output_icon} {prediction.replace('✅', '').replace('❌', '')}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Show score breakdown
            st.markdown("### 🔬 Score Breakdown")
            st.write(f"**🔁 Exact Match Score:** `{exact_score:.2f}`")
            if sim_stored_real is not None:
                st.write(f"**🧠 Similarity to Stored Real News:** `{sim_stored_real:.2f}`")
            if real_score is not None and fake_score is not None:
                st.write(f"**📈 Max Similarity with Real Dataset:** `{real_score:.2f}`")
                st.write(f"**📉 Max Similarity with Fake Dataset:** `{fake_score:.2f}`")
