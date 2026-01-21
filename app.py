import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
# Use this specific import to save memory
from keras.models import load_model

app = Flask(__name__)

# 1. FIX PATHS: Ensure these match your actual files on GitHub
# If they are in the root folder, remove "model/"
MODEL_PATH = "model\model.h5"
SCALER_PATH = "model\scaler.pkl"

model = None
scaler = None

# Load only once at startup
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # 2. CAPTURE DATA
            pclass = int(request.form["pclass"])
            sex = int(request.form["sex"])
            age = float(request.form["age"])
            sibsp = int(request.form["sibsp"])
            parch = int(request.form["parch"])
            fare = float(request.form["fare"])
            embarked = request.form["embarked"]

            # One-hot encoding for embarked (Matches your pd.get_dummies logic)
            # drop_first=True means we only need Q and S.
            embarked_Q = 1 if embarked == "Q" else 0
            embarked_S = 1 if embarked == "S" else 0

            # 3. FIX WARNING: Use a DataFrame to keep feature names for the scaler
            feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
            features_df = pd.DataFrame([[
                pclass, sex, age, sibsp, parch, fare, embarked_Q, embarked_S
            ]], columns=feature_cols)

            # Scale and Predict
            features_scaled = scaler.transform(features_df)
            probability = model.predict(features_scaled)[0][0]
            
            prediction = "Survived" if probability >= 0.5 else "Did Not Survive"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)