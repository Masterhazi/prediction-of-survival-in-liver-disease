import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Liver Survival Predictor", layout="centered")


st.title("Liver Cirrhosis Mortality Risk Predictor")
st.caption("Model trained using RandomForest with domain-based feature engineering and Child-Pugh scoring.")

model = joblib.load("liver_survival_pipeline.pkl")

# User Inputs
age = st.number_input("Age (in days)", value = None, min_value=1)
bilirubin = st.number_input("Bilirubin", value = None)
albumin = st.number_input("Albumin", value = None)
platelets = st.number_input("Platelets", value = None)
prothrombin = st.number_input("Prothrombin", value = None)
stage = st.number_input("Stage", value = None)

drug = st.selectbox("Drug", ["D-penicillamine", "Placebo"])
sex = st.selectbox("Sex", ["M", "F"])
ascites = st.selectbox("Ascites", ["Y", "N"])
hepatomegaly = st.selectbox("Hepatomegaly", ["Y", "N"])
spiders = st.selectbox("Spiders", ["Y", "N"])
edema = st.selectbox("Edema", ["N", "S", "Y"])

if st.button("Predict"):

    input_df = pd.DataFrame([{
        "id": 0,
        "N_Days": 0,
        "Drug": drug,
        "Age": age,
        "Sex": sex,
        "Ascites": ascites,
        "Hepatomegaly": hepatomegaly,
        "Spiders": spiders,
        "Edema": edema,
        "Bilirubin": bilirubin,
        "Cholesterol": None,
        "Albumin": albumin,
        "Copper": None,
        "Alk_Phos": None,
        "SGOT": None,
        "Tryglicerides": None,
        "Platelets": platelets,
        "Prothrombin": prothrombin,
        "Stage": stage,
        "Status": None
    }])

    prob = model.predict_proba(input_df)[0]

    if prob > 0.7:
        st.error("High Mortality Risk")
    elif prob > 0.4:
        st.warning("Moderate Risk")
    else:
        st.success("Lower Risk")


    st.subheader(f"Predicted Mortality Risk: {prob:.2%}")

