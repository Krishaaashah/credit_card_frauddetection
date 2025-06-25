from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import os
import joblib
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

app = Flask(__name__)
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
models = {
    "Logistic Regression": joblib.load("logistic_regression_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl"),
    "XGBoost": joblib.load("xgboost_model.pkl"),
    "LSTM": load_model("lstm_model.h5"),
    "Autoencoder": load_model("autoencoder_model.h5", compile=False)
}
models["Autoencoder"].compile(optimizer="adam", loss=MeanSquaredError())

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Descriptions and image names
model_descriptions = {
    "Logistic Regression": "Logistic Regression is a statistical method used to predict a binary outcome (like yes/no or 0/1) based on one or more input variables. It estimates the probability of the outcome using the logistic (sigmoid) function.",
    "Random Forest": "Random Forest is an ensemble learning method that builds multiple decision trees and combines their results to improve prediction accuracy and reduce overfitting. It works well for both classification and regression tasks.",
    "XGBoost": "A fast, accurate gradient boosting method used in many competitions.XGBoost (Extreme Gradient Boosting) is a powerful and efficient machine learning algorithm that builds an ensemble of decision trees using gradient boosting. It’s known for its high performance, speed, and accuracy in classification and regression tasks.",
    "LSTM": "LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) designed to learn and remember long-term dependencies in sequence data. It's especially useful for tasks like time series prediction, speech recognition, and natural language processing.",
    "Autoencoder": "Autoencoders are neural networks used for unsupervised learning that aim to compress input data into a lower-dimensional representation and then reconstruct it. They are commonly used for tasks like dimensionality reduction, denoising, and anomaly detection."
}

image_name_map = {
    "Logistic Regression": "logistic",
    "Random Forest": "rf",
    "XGBoost": "xgb",
    "LSTM": "lstm",
    "Autoencoder": "autoencoder"
}

@app.route('/')
def home():
    default_model = "Logistic Regression"
    short_name = image_name_map[default_model]
    return render_template(
        "index.html",
        model_names=list(models.keys()),
        selected_model=default_model,
        descriptions=model_descriptions,
        auc_image=f"auc_{short_name}.png",
        metrics_image=f"metrics_{short_name}.png"
    )

@app.route('/update_images', methods=['POST'])
def update_images():
    model_name = request.form.get("model")
    short_name = image_name_map[model_name]
    return {
        "auc_image": f"auc_{short_name}.png",
        "metrics_image": f"metrics_{short_name}.png",
        "description": model_descriptions[model_name]
    }

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form.get("model")
    file = request.files.get("file")
    if not file:
        return "No file uploaded!"

    df = pd.read_csv(file)
    X = scaler.transform(df)

    if model_name == "LSTM":
        X_lstm = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        y_scores = models[model_name].predict(X_lstm).flatten()
        y_pred = (y_scores > 0.5).astype(int)
    elif model_name == "Autoencoder":
        recon = models[model_name].predict(X)
        mse = np.mean(np.power(X - recon, 2), axis=1)
        threshold = np.percentile(mse, 95)
        y_pred = (mse > threshold).astype(int)
    else:
        y_pred = models[model_name].predict(X)

    df["Prediction"] = y_pred
    display_rows = df.head(100).to_dict(orient="records")

    # Fraud stats
    df["amt"] = df.get("amt", 0)
    fraud_count = int((df["Prediction"] == 1).sum())
    non_fraud_count = int((df["Prediction"] == 0).sum())
    total_fraud = float(df[df["Prediction"] == 1]["amt"].sum())
    avg_fraud = float(df[df["Prediction"] == 1]["amt"].mean() or 0)
    total_non_fraud = float(df[df["Prediction"] == 0]["amt"].sum())
    avg_non_fraud = float(df[df["Prediction"] == 0]["amt"].mean() or 0)

    short_name = image_name_map[model_name]
    return render_template(
        "result.html",
        model_name=model_name,
        display_rows=display_rows,
        fraud_count=fraud_count,
        non_fraud_count=non_fraud_count,
        total_fraud=round(total_fraud, 2),
        avg_fraud=round(avg_fraud, 2),
        total_non_fraud=round(total_non_fraud, 2),
        avg_non_fraud=round(avg_non_fraud, 2),
        auc_image=f"auc_{short_name}.png",
        metrics_image=f"metrics_{short_name}.png"
    )

if __name__ == '__main__':
    app.run(debug=True)
