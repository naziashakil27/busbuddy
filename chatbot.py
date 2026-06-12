import random
import json
import pickle
import numpy as np
import nltk
import pandas as pd
import re
import os

from fuzzywuzzy import process
from flask import Flask, render_template, request, jsonify
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Download NLTK data at startup (required on cloud environments like Render)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ---------------- Flask App ----------------
app = Flask(__name__)

# ---------------- Load Data ----------------
bus_data = pd.read_csv("surat_bus.csv")
df = pd.read_csv("SURAT5.csv")

# Pre-convert stop names to clean, lowercase strings to prevent NaN crashes on startup
df["originStopName"] = df["originStopName"].dropna().astype(str).str.lower().str.strip()
df["destinationStopName"] = df["destinationStopName"].dropna().astype(str).str.lower().str.strip()

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.keras")

with open("fare_prediction_model.pkl", "rb") as file:
    fare_model = pickle.load(file)

# ---------------- Fare Prediction ----------------
def predict_fare(origin, destination):
    origin = origin.lower().strip()
    destination = destination.lower().strip()

    # Query directly from pre-loaded and sanitized df in memory to save disk I/O
    row = df[
        (df["originStopName"] == origin) &
        (df["destinationStopName"] == destination)
    ]

    if not row.empty:
        return row["fareForChild"].values[0], row["fareForAdult"].values[0]

    return None, None

# ---------------- NLP ----------------
def clean_up_sentence(sentence):
    return [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)]

def bag_of_words(sentence):
    bag = [0] * len(words)
    sentence_words = clean_up_sentence(sentence)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        return [{"intent": "no_match"}]

    return [{"intent": classes[r[0]]} for r in results]

# ---------------- Stop Matching ----------------
combined_list = (
    df["originStopName"].tolist() +
    df["destinationStopName"].tolist()
)

def extract_origin_destination(message):
    match = re.search(r"from\s+(.*?)\s+to\s+(.*)", message.lower())
    if not match:
        return None, None

    origin, destination = match.groups()
    origin = origin.strip()
    destination = destination.strip()
    if not origin or not destination:
        return None, None

    o = process.extractOne(origin, combined_list) if origin else None
    d = process.extractOne(destination, combined_list) if destination else None

    return o[0] if o and o[1] > 80 else None, d[0] if d and d[1] > 80 else None

# ---------------- Bus Routes ----------------
def find_direct_buses(start, end):
    result = []
    for _, row in bus_data.iterrows():
        stops = row[1:].dropna().str.lower().tolist()
        if start in stops and end in stops:
            result.append(row["bus no"])
    return result

# ---------------- Chat Response ----------------
def get_response(message):
    ints = predict_class(message)
    tag = ints[0]["intent"]

    if tag == "fare":
        origin, destination = extract_origin_destination(message)
        if not origin or not destination:
            return "Please specify the starting and destination stops (e.g., 'fare from Adajan to Vesu')."

        c, a = predict_fare(origin, destination)
        if c is None:
            return f"Sorry, fare information between '{origin or 'unknown stop'}' and '{destination or 'unknown stop'}' was not found in our database."
        return f"The bus fare from {origin.title()} to {destination.title()} is ₹{c} for children and ₹{a} for adults."

    if tag == "route_availability":
        origin, destination = extract_origin_destination(message)
        if not origin or not destination:
            return "Please specify the starting and destination stops (e.g., 'buses from Adajan to Vesu')."
        buses = find_direct_buses(origin, destination)
        if buses:
            return f"Direct buses available from {origin.title()} to {destination.title()}: {', '.join(map(str, buses))}"
        return f"No direct buses were found between {origin.title()} and {destination.title()}."

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn’t understand that. You can ask me about bus fares or direct routes between stops!"

# ---------------- Flask Routes ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message")
    reply = get_response(msg)
    return jsonify({"response": reply})

# ---------------- Run ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
