from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text
def preprocess_text(text):
    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word.isalpha()
    ]

    return " ".join(tokens)
app = FastAPI()
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
class Review(BaseModel):
    text: str

@app.post("/predict")
def predict(review: Review):
    processed_text = preprocess_text(clean_text(review.text))
    text_features = tfidf.transform([processed_text])
    prediction = model.predict(text_features)[0]
    return {"prediction": int(prediction)}
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>IMDb Sentiment Predictor</title>
    </head>
    <body>
        <h1>IMDb Sentiment Predictor</h1>
        <textarea id="review" rows="6" cols="50" placeholder="Enter a movie review"></textarea>
        <br><br>
        <button onclick="predict()">Predict</button>
        <h2 id="result"></h2>

        <script>
        async function predict() {
            const text = document.getElementById("review").value;

            const response = await fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({text: text})
            });

            const data = await response.json();
            document.getElementById("result").innerText =
                data.prediction === 1 ? "Positive" : "Negative";
        }
        </script>
    </body>
    </html>
    """