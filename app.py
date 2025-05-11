from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipe import CustomData, PredictPipeline

application = Flask(__name__)

app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    else:
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethinicity=request.form.get("race/ethnicity"),
            parental_level_of_education=request.form.get("parental level of education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test preparation course"),
            reading_score=request.form.get("reading score"),
            writing_score=request.form.get("writing score"),
        )
        pred_df = data.get_data_as_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        return render_template("home.html", prediction=result[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
