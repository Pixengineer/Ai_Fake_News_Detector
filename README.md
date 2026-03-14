`# 📰 AI Fake News Detector

**AI Fake News Detector** is a Machine Learning web application that detects whether a news article is **Real or Fake** using Natural Language Processing (NLP) techniques.

The model analyzes news content and predicts its authenticity based on patterns learned from a labeled dataset of real and fake news articles.

The application is deployed using **Streamlit Cloud**, allowing users to test the model directly through a web interface.

---

## 🚀 Live Demo

🔗 **Live App:**
https://aifakenewsdetectorbysatyam.streamlit.app

---

## 📌 Features

* 🔍 Detect whether a news article is **Real or Fake**
* 📄 **Paste full news text** for analysis
* 🌐 **Enter a news URL** to automatically extract article content
* 📊 **Confidence score** showing prediction certainty
* 📈 **Progress bar visualization**
* 🎨 Clean and interactive **Streamlit UI**

---

## 🧠 Machine Learning Pipeline

The model follows a standard NLP pipeline:

1. **Text Preprocessing**

   * Lowercasing
   * Removing special characters
   * Stopword removal

2. **Feature Extraction**

   * TF-IDF Vectorization

3. **Model Training**

   * Logistic Regression classifier

4. **Prediction**

   * Fake / Real classification
   * Confidence probability score

---

## 📊 Model Performance

The trained model achieves approximately:

**Accuracy: ~98%**

Evaluation techniques used:

* Train-Test Split
* Confusion Matrix
* Classification Report

---

## 🛠️ Tech Stack

**Programming Language**

* Python

**Machine Learning & NLP**

* Scikit-learn
* NLTK
* TF-IDF Vectorizer

**Data Processing**

* Pandas
* NumPy

**Visualization**

* Matplotlib
* Seaborn

**Web Application**

* Streamlit

**Article Extraction**

* Newspaper3k

---

## 📂 Project Structure

```
AI_Fake_News_Detector
│
├── app.py                  # Streamlit web application
├── fake_news_detection.py  # Model training script
├── fake_news_model.pkl     # Trained ML model
├── vectorizer.pkl          # TF-IDF vectorizer
├── requirements.txt        # Project dependencies
└── dataset/
    ├── Fake.csv
    └── True.csv
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Pixengineer/AI_Fake_News_Detector.git
```

Navigate to the project folder:

```
cd AI_Fake_News_Detector
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the Streamlit application:

```
streamlit run app.py
```

---

## 🧪 How to Use

1. Open the web app
2. Choose input method:

   * Paste News Text
   * Enter News URL
3. Click **Predict**
4. View the result:

   * Fake or Real
   * Confidence Score

---

## 📈 Future Improvements

Possible upgrades for the project:

* Transformer models (BERT / DistilBERT)
* Explainable AI for prediction reasoning
* News credibility scoring
* Real-time news API integration
* Advanced UI/UX improvements

---

## 👨‍💻 Author

**Satyam Bhardwaj**

B.Tech – Mathematics & Computing
Rajiv Gandhi Institute of Petroleum Technology (RGIPT)

---

## ⭐ Support

If you found this project useful, consider giving it a **star ⭐ on GitHub**.
