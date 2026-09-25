"""
Streamlit app - Wellness Tourism Package purchase prediction ("Visit with Us").
Loads the model committed to this folder by the GitHub Actions pipeline,
collects customer inputs into a DataFrame and returns a purchase prediction.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "best_tourism_model_v1.joblib"
THRESHOLD = 0.5


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Wellness Package Predictor", layout="wide")
st.title("Visit with Us - Wellness Tourism Package Predictor")
st.write("Enter the customer's details to estimate whether they will purchase "
         "the Wellness Tourism Package before the sales team contacts them.")

model = load_model()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer profile")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly income", min_value=0.0, max_value=500000.0,
                                     value=23000.0, step=500.0)
    city_tier = st.selectbox("City tier", [1, 2, 3])

with col2:
    st.subheader("Travel preferences")
    num_persons = st.number_input("Number of persons visiting", min_value=1, max_value=10, value=3)
    num_children = st.number_input("Number of children (<5 yrs) visiting", min_value=0, max_value=5, value=1)
    preferred_star = st.selectbox("Preferred property star", [3, 4, 5])
    num_trips = st.number_input("Average trips per year", min_value=0, max_value=30, value=3)
    passport = st.selectbox("Has a valid passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns a car?", ["No", "Yes"])

with col3:
    st.subheader("Sales interaction")
    type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Duration of pitch (minutes)", min_value=0, max_value=180, value=15)
    num_followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4)
    pitch_score = st.slider("Pitch satisfaction score", min_value=1, max_value=5, value=3)

# Collect inputs into a single-row DataFrame with the training column names
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

with st.expander("Input data sent to the model"):
    st.dataframe(input_df)

if st.button("Predict", type="primary"):
    proba = float(model.predict_proba(input_df)[0, 1])
    if proba >= THRESHOLD:
        st.success(f"Likely to PURCHASE the Wellness package (probability {proba:.1%}). "
                   "Prioritise this customer for outreach.")
    else:
        st.warning(f"Unlikely to purchase (probability {proba:.1%}). "
                   "Lower priority for the Wellness campaign.")
