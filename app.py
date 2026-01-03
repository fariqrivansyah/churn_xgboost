from flask import Flask, render_template, request
import pandas as pd
import joblib
import os


app = Flask(__name__)

# =====================
# LOAD MODEL & FEATURE
# =====================
model = joblib.load("xgb_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# =====================
# HOME
# =====================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        # Ambil input dari form
        input_data = {
            "gender": request.form["gender"],
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": int(request.form["tenure"]),
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"]),
            "Contract": request.form["Contract"],
            "InternetService": request.form["InternetService"],
            "PaymentMethod": request.form["PaymentMethod"]
        }

        # Buat DataFrame
        df_input = pd.DataFrame([input_data])

        # One Hot Encoding
        df_input = pd.get_dummies(df_input)

        # Samakan fitur dengan training
        df_input = df_input.reindex(columns=feature_names, fill_value=0)

        # Prediksi
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        result = "Churn" if prediction == 1 else "Tidak Churn"

        return render_template(
            "result.html",
            result=result,
            probability=round(probability * 100, 2)
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )

