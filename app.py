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

def download_and_extract_model():
    if not os.path.exists("model"):
        file_id = "1vQkKsrnKFwE7VNzKYfDrOaOtWNr1CL5C"  # 🔁 Replace with your actual file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "model.zip"

        gdown.download(url, output, quiet=False)

        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(".")  # Extracts 'model/' folder

        os.remove(output)

download_and_extract_model()

# Page config
st.set_page_config(
    page_title="📰 Fake News Detector",
    layout="centered",
)

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

# Set background here
set_background("wallpaper1.jpg")  # Make sure 'wallpaper.png' is in the same directory




# Custom CSS for background and textbox animation
st.markdown(
    """
    <style>
        /* Container background blur */
        .main {
            background-color: rgba(0, 0, 0, 0.7);
            padding: 2rem;
            border-radius: 10px;
        }

        /* Gradient animated title */
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



        @keyframes animateGradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Tilt effect on input label */
        .textbox-label {
            display: inline-block;
            transform-style: preserve-3d;
            transition: transform 0.2s ease;
            color: #e8faf9; 
        }
        .textbox-label:hover {
            transform: perspective(500px) rotateY(5deg) rotateX(5deg);
        }

        /* Animated Predict Button */
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

/* Hover effect for the whole container */
div[data-baseweb="textarea"]:hover {
    transform: scale(1.01);
    box-shadow: 0 6px 20px rgba(255, 187, 108, 1);
    border-color: #ffbb6c;
    cursor: pointer;
}

/* Inner text style (optional, keep if needed) */
textarea {
    color: white;
    background-color: transparent;
}
    </style>
    """,
    unsafe_allow_html=True
)



# Title
st.markdown('<div class="title">🧠 Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="textbox-label"><h4>💬 Enter a news article below to check if it\'s Real or Fake</h4></div>', unsafe_allow_html=True)
st.markdown('<div class="textbox-label"> <h4>✍️ Type or paste the news article here(Dataset):</h4></div>', unsafe_allow_html=True)

# Load files
@st.cache_resource
def load_resources():
    model = BertForSequenceClassification.from_pretrained("model").to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("model")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    real_embeddings = np.load("real_embeddings.npy")
    fake_embeddings = np.load("fake_embeddings.npy")

    with open("exact_match_dict.pkl", "rb") as f:
        exact_match_dict = pickle.load(f)

    return model, tokenizer, sbert_model, real_embeddings, fake_embeddings, exact_match_dict

model, tokenizer, sbert_model, real_embeddings, fake_embeddings, exact_match_dict = load_resources()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prediction function
def predict_news(text):
    text = text.strip()

    # 1. Exact match check
    if text in exact_match_dict:
        return "✅ Real" if exact_match_dict[text] == 1 else "❌ Fake"

    # 2. Semantic similarity check
    input_embedding = sbert_model.encode(text, convert_to_tensor=True).to(device)
    real_tensor = torch.tensor(real_embeddings).to(device)
    fake_tensor = torch.tensor(fake_embeddings).to(device)

    real_sim = util.cos_sim(input_embedding, real_tensor).max().item()
    fake_sim = util.cos_sim(input_embedding, fake_tensor).max().item()

    if abs(real_sim - fake_sim) > 0.05:
        return "✅ Real News" if real_sim > fake_sim else "❌ Fake News"

    # 3. Classifier fallback
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs).item()
    return "✅ Real News" if label == 1 else "❌ Fake News"

# Input
with st.container():
    user_input = st.text_area("",height=200, key="textbox", help="Enter the news to check fake or real",placeholder=" Please Enter The Input Here...")

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
            {output_icon} {"Please enter some text to predict."}
        </span>
    </div>
    """,unsafe_allow_html=True)

    else:
        with st.spinner("Analyzing..."):
            prediction = predict_news(user_input)
            output_color = "#00cc66" if "Real" in prediction else "#ff4d4d"
            output_icon = "✅" if "Real" in prediction else "❌"
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
    """,unsafe_allow_html=True)

