from flask import Flask, request, jsonify, render_template
import joblib
import os
import re

app = Flask(__name__)

# Load model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vectorizer = joblib.load(os.path.join(BASE_DIR, "svm_vectorizer_word1_4000.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "svm_model_word1_4000.pkl"))

# Hardcoded stopwords
stop_words = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","any","both","each","few","more",
    "most","other","some","such","no","nor","not","only","own","same","so",
    "than","too","very","can","will","just","don","should","now"
}

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("user_input", "").strip()

    cleaned = clean_text(user_input)
    transformed = vectorizer.transform([cleaned])
    pred = model.predict(transformed)[0]

    sentiment = "Positive" if pred == 2 else "Negative"
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run()
