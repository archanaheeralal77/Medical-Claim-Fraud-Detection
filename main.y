from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

# Load saved model, scaler, encoders
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Initialize FastAPI app
app = FastAPI(title="Medical Claim Fraud Detection API")

# Define input schema using Pydantic
class ClaimData(BaseModel):
    provider_id: str
    procedure_code: str
    billing_frequency: str
    patient_id: str
    total_amount: float
    doctor: str
    hospital: str
    diagnosis_report: str
    discharge_summary: str
    prescriptions_and_bills: str
    hospital_bills: float
    insurance_company: str
    contact_details: str
    network_partners: str
    benefits: str
    policy_number: str
    policy_type: str
    renewal_date: str
    claim_limits: float
    premium_amount: float
    start_date: str
    end_date: str
    bank_account: str
    policy_number_claim: str
    policy_holder_name: str
    address: str
    phone_number: str
    email: str
    covered: str
    hospitalized_date: str
    treatment_expenses: float
    claim_documents_submitted: str

@app.post("/predict/")
def predict_fraud(claim: ClaimData):
    # Convert to DataFrame
    data = pd.DataFrame([claim.dict()])

    # Encode categorical columns
    for col in data.columns:
        if col in label_encoders:
            le = label_encoders[col]
            try:
                data[col] = le.transform(data[col].astype(str))
            except:
                # Handle unseen labels
                data[col] = le.transform([le.classes_[0]])

    # Scale
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)[0]
    result = "Fraud" if prediction == 1 else "Not Fraud"
    return {"prediction": int(prediction), "result": result}
