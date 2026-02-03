from flask import Flask,jsonify,request,render_template
import pickle
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
application=Flask(__name__)
app=application


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")

model = pickle.load(open(model_path, "rb"))

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get form data
        Brand = request.form.get("Brand")
        Model = request.form.get("Model")
        Seller_Type = request.form.get("Seller_Type")
        Owner = request.form.get("Owner")
        KM_Driven = float(request.form.get("KM_Driven"))
        Ex_Showroom_Price = float(request.form.get("Ex_Showroom_Price"))

        # Transform data and predict
        input_df = pd.DataFrame([{
        "Brand": Brand,
        "Model": Model,
        "Seller_Type": Seller_Type,
        "Owner": Owner,
        "KM_Driven": KM_Driven,
        "Ex_Showroom_Price": Ex_Showroom_Price
    }])

        result = model.predict(input_df)

        return render_template("home.html", results=round(result[0], 2))
    else:
        return render_template('home.html')  # fixed template name


if __name__=="__main__":
    app.run(host='0.0.0.0')
    