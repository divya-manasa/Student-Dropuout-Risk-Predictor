from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "model_dropout_rf.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    model = None
    print(f"Error: Model file not found at {MODEL_PATH}")

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/input')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Model not loaded", 500

    try:
        # Get data from Form (not JSON)
        login_frequency = float(request.form.get('login_frequency', 0))
        time_spent_hours = float(request.form.get('time_spent_hours', 0))
        assignments_completed = float(request.form.get('assignments_completed', 0))
        quiz_score = float(request.form.get('quiz_score', 0))
        days_inactive = float(request.form.get('days_inactive', 0))

        # Derived features logic
        engagement_score = login_frequency * time_spent_hours
        performance_index = 0.7 * assignments_completed + 0.3 * quiz_score
        inactivity_flag = 1 if days_inactive >= 14 else 0

        # Create input DataFrame
        input_data = pd.DataFrame([{
            "login_frequency": login_frequency,
            "time_spent_hours": time_spent_hours,
            "assignments_completed": assignments_completed,
            "quiz_score": quiz_score,
            "days_inactive": days_inactive,
            "engagement_score": engagement_score,
            "performance_index": performance_index,
            "inactivity_flag": inactivity_flag
        }])

        # Predict
        prob = model.predict_proba(input_data)[0][1]

        # Determine risk category
        if prob < 0.30:
            risk = "Low Risk"
            risk_class = "low-risk"
        elif prob < 0.60:
            risk = "Medium Risk"
            risk_class = "medium-risk"
        else:
            risk = "High Risk"
            risk_class = "high-risk"

        probability_pct = round(prob * 100, 1)

        return render_template('result.html', 
                               probability_pct=probability_pct, 
                               risk=risk, 
                               risk_class=risk_class)

    except Exception as e:
        return f"An error occurred: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)
