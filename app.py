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

# --- Page config ---
st.set_page_config(
    page_title="📰 Fake News Detector",
    layout="centered",
)

# --- Background image ---
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
st.markdown('<div class="textbox-label"> <h4>✍️ Type or paste the news article here(Dataset):</h4></div>', unsafe_allow_html=True)
# File paths for persistent storage
REAL_NEWS_STORE = "stored_real_news.pkl"

# Load or initialize real news storage
def load_stored_real_news():
    if os.path.exists(REAL_NEWS_STORE):
        with open(REAL_NEWS_STORE, "rb") as f:
            return pickle.load(f)
    return []

def save_stored_real_news(news_list):
    with open(REAL_NEWS_STORE, "wb") as f:
        pickle.dump(news_list, f)

stored_real_news = load_stored_real_news()

# Update predict_news to include real-news tracking
def predict_news(text):
    text = text.strip()

    # 1. Exact match check
    if text in exact_match_dict:
        return "✅ Real", True

    # 2. Compare with stored real news to detect fake modification
    input_embedding = sbert_model.encode(text, convert_to_tensor=True).to(device)
    is_fake_due_to_change = False

    for real_news in stored_real_news:
        real_embedding = sbert_model.encode(real_news, convert_to_tensor=True).to(device)
        sim = util.cos_sim(input_embedding, real_embedding).item()
        if sim >= 0.8:
            is_fake_due_to_change = True
            break

    if is_fake_due_to_change:
        return "❌ Fake (Modified from Real)", False

    # 3. Semantic similarity with original real/fake dataset
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
            return "✅ Real News", True
        else:
            return "❌ Fake News", False

    # 4. Classifier fallback
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs).item()

    if label == 1:
        if text not in stored_real_news:
            stored_real_news.append(text)
            save_stored_real_news(stored_real_news)
        return "✅ Real News", True
    else:
        return "❌ Fake News", False

# Input
with st.container():
    user_input = st.text_area("", height=200, key="textbox", help="Enter the news to check fake or real", placeholder=" Please Enter The Input Here...")

# Predict
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        output_color = "#ffb02e"
        output_icon = "⚠️"
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
                {output_icon} Please enter some text to predict.
            </span>
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Analyzing..."):
            prediction, is_real = predict_news(user_input)
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
