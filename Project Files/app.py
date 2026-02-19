from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
try:
    model = pickle.load(open("best_fraud_model.pkl", "rb"))
except FileNotFoundError:
    print("Error: best_fraud_model.pkl not found. Please ensure the model file is in the root directory.")


@app.route("/")
def index():
    # This route now only shows the project details/landing page
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    output = None

    if request.method == "POST":
        # Inputs from predict.html
        step = int(request.form["step"])
        tx_type = int(request.form["type"])
        amount = float(request.form["amount"])
        old_org = float(request.form["oldbalanceOrg"])
        new_org = float(request.form["newbalanceOrig"])
        old_dest = float(request.form["oldbalanceDest"])
        new_dest = float(request.form["newbalanceDest"])

        # Feature engineering (Must match the training features)
        error_org = new_org + amount - old_org
        error_dest = old_dest + amount - new_dest

        X = np.array([[step, tx_type, amount,
                       old_org, new_org,
                       old_dest, new_dest,
                       error_org, error_dest]])

        # Get probability for the "Fraud" class
        confidence = round(model.predict_proba(X)[0][1] * 100, 2)

        # 🔐 Decision Logic
        if confidence <= 20:
            label = "✅ Legitimate Transaction"
            risk = "🟢 Low Risk"
            meaning = "This transaction follows normal user behavior."
            action = "Transaction approved instantly (No friction for the user)."

        elif confidence <= 70:
            label = "⚠️ Potentially Fraudulent Transaction"
            risk = "🟡 Medium Risk"
            meaning = "This transaction shows unusual or suspicious behavior."
            action = "Transaction held. Phone call or SMS OTP verification required."

        else:
            label = "❌ Fraudulent Transaction"
            risk = "🔴 High Risk"
            meaning = "This transaction strongly matches known fraud patterns."
            action = "Transaction blocked immediately and the account is frozen."

        output = {
            "label": label,
            "confidence": confidence,
            "risk": risk,
            "meaning": meaning,
            "action": action
        }

    # Render predict.html instead of index.html
    return render_template("predict.html", output=output)


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


if __name__ == "__main__":
    app.run(debug=True)
