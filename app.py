from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from src.mlproject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "Training Successful"

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            # Reading inputs from form
            age = int(request.form['age'])
            sex = request.form['sex']       # 'male' or 'female'
            bmi = float(request.form['bmi'])
            children = int(request.form['children'])
            smoker = request.form['smoker'] # 'yes' or 'no'
            region = request.form['region'] # 'southwest', etc.

            # Creating dict for DataFrame
            data = {
                'age': [age],
                'sex': [sex],
                'bmi': [bmi],
                'children': [children],
                'smoker': [smoker],
                'region': [region]
            }

            obj = PredictionPipeline()
            transformed = obj.transform(data)
            prediction = obj.predict(transformed)

            return render_template('results.html', prediction=round(prediction[0], 2))

        except Exception as e:
            print("The Exception Message is:", e)
            return 'Something went wrong!'

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
