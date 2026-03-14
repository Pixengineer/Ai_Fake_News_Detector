import streamlit as st
import pickle
import os
from newspaper import Article

# Page settings
st.set_page_config(page_title="AI Fake News Detector", layout="centered")

# Load model
BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR,"fake_news_model.pkl"),"rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR,"vectorizer.pkl"),"rb"))

# Title
st.title("📰 AI Fake News Detector")

st.markdown(
"""
This application uses **Machine Learning (NLP)** to detect whether a news article is **Real or Fake**.
You can either paste the **news text** or provide a **news URL**.
"""
)

st.divider()

# Input selection
option = st.radio(
    "Choose Input Type",
    ("Paste News Text", "Enter News URL")
)

# Function to extract news from URL
def extract_news_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

news_text = ""

# Text input
if option == "Paste News Text":

    news_text = st.text_area(
        "Paste the news article here",
        height=200
    )

# URL input
else:

    url = st.text_input("Enter News URL")

    if st.button("Fetch News"):
        try:
            news_text = extract_news_from_url(url)
            st.success("News fetched successfully")
            st.write(news_text[:1500])
        except:
            st.error("Could not fetch article from this URL")

# Prediction
if st.button("Predict"):

    if news_text.strip() == "":
        st.warning("Please enter news text first")
    else:

        news_vec = vectorizer.transform([news_text])

        prediction = model.predict(news_vec)

        probability = model.predict_proba(news_vec)

        confidence = max(probability[0]) * 100

        st.divider()

        if prediction[0] == 0:
            st.error("🚨 Fake News Detected")
        else:
            st.success("✅ Real News")

        st.write(f"### Confidence Score: {confidence:.2f}%")

        st.progress(int(confidence))