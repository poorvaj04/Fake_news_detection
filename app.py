import streamlit as st
import re
import nltk
import pickle
import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load punkt tokenizer
punkt_path = r"D:\\naan_mudhalvan\\app\\tokenizers\\punkt\\english.pickle"
with open(punkt_path, 'rb') as f:
    sentence_tokenizer = pickle.load(f)

# Load TF-IDF vectorizer and model (Assuming you save them beforehand)
model = joblib.load(r"D:\\naan_mudhalvan\\app\\logistic_model.pkl")
tfidf_vectorizer = joblib.load(r"D:\\naan_mudhalvan\\app\\tfidf_vectorizer.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    sentences = sentence_tokenizer.tokenize(text)
    words = []
    for sent in sentences:
        words.extend(sent.split())
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def predict_news(text):
    cleaned = clean_text(text)
    vect = tfidf_vectorizer.transform([cleaned])
    prediction = model.predict(vect)
    prob = model.predict_proba(vect)[0]
    if prediction[0] == 0:
        return f"🟢 Real News (Confidence: {prob[0]*100:.2f}%)"
    else:
        return f"🔴 Fake News (Confidence: {prob[1]*100:.2f}%)"

# Streamlit UI
st.title("📰 Fake News Detection App")
user_input = st.text_area("Enter a news headline or article text:")
if st.button("Analyze"):
    if user_input.strip():
        result = predict_news(user_input)
        st.success(result)
    else:
        st.warning("Please enter some text.")
