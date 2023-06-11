#Creating the web app with streamlit
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


def predict_quality(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    
    return predictions_data['Label'][0]
    
model = load_model('diabetes_deployment_model')

st.title("Diabetes Predictive Framework")
st.write('This framework is created to predict diabetes in hospital clients by Okeke')

Pregnancies = st.sidebar.slider('Pregnancies', 
                                min_value=0.00, 
                                max_value=20.00, value=0.00, 
                                step = 1.00)

Glucose = st.number_input('Glucose', 
                          min_value=0.00, 
                          max_value=250.00, 
                          value=50.00)

BloodPressure = st.number_input('BloodPressure', 
                                min_value=0.00, 
                                max_value=200.00, 
                                value=120.00)

SkinThickness = st.number_input('SkinThickness', 
                                min_value=0.00, 
                                max_value=100.00, 
                                value=60.00)

Insulin = st.number_input('Insulin', 
                          min_value=0.00, 
                          max_value=1000.00, 
                          value=200.00)

BMI = st.number_input('BMI', 
                      min_value=0.00, 
                      max_value=100.00, 
                      value=50.00)

DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', 
                                           min_value=0.01, 
                                           max_value=10.00, 
                                           value=1.00) 

Age = st.sidebar.slider('Age', 
                        min_value=1.00, 
                        max_value=150.00, 
                        value=60.00, 
                        step = 1.00)


features = {'Pregnancies': Pregnancies, 'Glucose': Glucose,
            'BloodPressure': BloodPressure, 'SkinThickness': SkinThickness,
            'Insulin': Insulin, 'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction, 'Age': Age,
            }
 

features_df  = pd.DataFrame([features])

st.table(features_df)  

if st.button('Predict'):
    
    prediction = predict_quality(model, features_df)
    
    st.write('Based on values provided, the patient is '+ str(prediction))