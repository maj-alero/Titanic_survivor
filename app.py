from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("model/model.h5")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        pclass = int(request.form["pclass"])
        sex = int(request.form["sex"])
        age = float(request.form["age"])
        sibsp = int(request.form["sibsp"])
        parch = int(request.form["parch"])
        fare = float(request.form["fare"])
        embarked = request.form["embarked"]

        # One-hot encoding for embarked
        embarked_C = 1 if embarked == "C" else 0
        embarked_Q = 1 if embarked == "Q" else 0

        features = np.array([[
            pclass, sex, age, sibsp, parch, fare, embarked_C, embarked_Q
        ]])

        features = scaler.transform(features)

        probability = model.predict(features)[0][0]
        prediction = "Survived" if probability >= 0.5 else "Did Not Survive"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
