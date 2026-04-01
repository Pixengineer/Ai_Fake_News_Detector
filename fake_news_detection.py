import pandas as pd
print("Program started")
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


nltk.download('stopwords')


fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")


fake["label"] = 0
true["label"] = 1


data = pd.concat([fake, true])
data = data.sample(frac=1)


data = data[["text", "label"]]


stop_words = set(stopwords.words("english"))


def clean_text(text):

    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)


data["text"] = data["text"].apply(clean_text)


X = data["text"]
y = data["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression()


model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)


accuracy = accuracy_score(y_test, y_pred)

print("Model training completed")
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6,4))

sns.heatmap(cm, annot=True, fmt='d',
            cmap='Blues',
            xticklabels=['Fake','Real'],
            yticklabels=['Fake','Real'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


import pickle

pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully")


news = ["Government announces new economic reforms"]

news_vec = vectorizer.transform(news)

prediction = model.predict(news_vec)

if prediction[0] == 0:
    print("Fake News")
else:
    print("Real News")





