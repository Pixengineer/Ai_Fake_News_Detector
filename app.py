import streamlit as st
import pickle
import os
from newspaper import Article


st.set_page_config(page_title="AI Fake News Detector", layout="centered")


BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "fake_news_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))


st.title("📰 AI Fake News Detector")
st.caption("Developed by Satyam Bhardwaj")

st.markdown(
"""
Detect whether a news article is **Real or Fake** using Machine Learning and NLP.
You can either **paste the news text** or **provide a news URL**.
"""
)

st.divider()


option = st.radio(
    "Choose Input Type",
    ("Paste News Text", "Enter News URL")
)


def extract_news_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

news_text = ""


if option == "Paste News Text":
    news_text = st.text_area(
        "Paste the news article here",
        height=200
    )


else:
    url = st.text_input("Enter News URL")

    if st.button("Fetch News"):
        try:
            news_text = extract_news_from_url(url)
            st.success("News fetched successfully")
            st.write(news_text[:1500])
        except:
            st.error("Could not fetch article from this URL")


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


st.markdown("---")
st.markdown("Made with ❤️ by **Satyam Bhardwaj**")
