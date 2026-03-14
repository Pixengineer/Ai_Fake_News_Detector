import streamlit as st
import pickle
import os

BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR,"fake_news_model.pkl"),"rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR,"vectorizer.pkl"),"rb"))

st.title("AI Fake News Detector")

news = st.text_area("Enter News Text")

if st.button("Predict"):

    news_vec = vectorizer.transform([news])

    prediction = model.predict(news_vec)

    if prediction[0] == 0:
        st.error("Fake News")
    else:
        st.success("Real News")