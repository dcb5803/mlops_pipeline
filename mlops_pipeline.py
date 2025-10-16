import os, time, joblib, logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from flask import Flask, request, jsonify
import mlflow

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
MODEL_PATH = "model.joblib"

def load_data():
    data = load_iris()
    return pd.DataFrame(data.data, columns=data.feature_names), data.target

def train_model():
    X, y = load_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    mlflow.sklearn.log_model(model, "rf_model")
    logging.info("✅ Model trained and saved.")
    return model

def predict(model, features):
    return model.predict([features])[0]

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    model = joblib.load(MODEL_PATH)
    features = request.json["features"]
    pred = predict(model, features)
    return jsonify({"prediction": int(pred)})

@app.route("/retrain", methods=["POST"])
def retrain_endpoint():
    train_model()
    return jsonify({"status": "Model retrained"})

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH): train_model()
    app.run(host="0.0.0.0", port=8080)
