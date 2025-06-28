# **Fake News Detection System Using NLP**
This project is an interactive and visually enhanced web application that helps to identify whether a given news article is Real or Fake(based on dataset). It provides accurate predictions and a user-friendly interface, powered by cutting-edge Natural Language Processing (NLP) models like BERT for classification and SBERT for semantic similarity.
## **Datasets**
Link for the App.py:
[WEB_APP](https://fakenewsdetection-nnq2kg4wb5vbmerwuvukos.streamlit.app/)
## **How It Works**
1. The input text is first checked against a dictionary of known articles.

2. If no exact match is found, semantic similarity is calculated using SBERT embeddings.

3. If still uncertain, a fine-tuned BERT model classifies the input as Real or Fake.

4. Results are displayed in an attractive styled alert box with icons and color codes.
