from flask import Flask, render_template, request
import pickle
import pandas as pd
app = Flask(__name__)

# ✅ Load the full pipeline (preprocessing + model)
with open("model_churn_pikle.pkl", "rb") as f:
    model = pickle.load(f)


with open("process_churn_pikle.pkl", "rb") as f:
    process = pickle.load(f)



@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Extract input values
        input_data = {
            "gender": request.form["gender"],
            "SeniorCitizen": request.form["SeniorCitizen"],
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": float(request.form["tenure"]),
            "PhoneService": request.form["PhoneService"],
            "MultipleLines": request.form["MultipleLines"],
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "OnlineBackup": request.form["OnlineBackup"],
            "DeviceProtection": request.form["DeviceProtection"],
            "TechSupport": request.form["TechSupport"],
            "StreamingTV": request.form["StreamingTV"],
            "StreamingMovies": request.form["StreamingMovies"],
            "Contract": request.form["Contract"],
            "PaperlessBilling": request.form["PaperlessBilling"],
            "PaymentMethod": request.form["PaymentMethod"],
            
            
            "MonthlyCharges": float(request.form["MonthlyCharges"]),

            "TotalCharges": float(request.form["TotalCharges"]),


        }

        df = pd.DataFrame([input_data])


        
        x_train_processed = process.transform(df)
        

    
        prediction = model.predict(  x_train_processed)
        if prediction==1:
            prediction="Customer churn is highly likely."
        elif prediction==0:     
            prediction= "Customer churn is unlikely."
        else :
            prediction  = 'pls fill all values'



        print(prediction)

        return render_template("index.html", prediction_text=f"Prediction: {prediction}")
    
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
