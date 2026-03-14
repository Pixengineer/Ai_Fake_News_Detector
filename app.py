import streamlit as st
import pickle

# load model
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection System")

news = st.text_area("Enter News Text")

if st.button("Predict"):

    news_vec = vectorizer.transform([news])

    prediction = model.predict(news_vec)

    if prediction[0] == 0:
        st.error("Fake News ❌")
    else:
        st.success("Real News ✅")