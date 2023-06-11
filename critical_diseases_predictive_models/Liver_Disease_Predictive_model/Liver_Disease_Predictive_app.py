#Creating the web app with streamlit
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


def predict_quality(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    
    return predictions_data['Label'][0]
    
model = load_model('liver_disease_deployment_model')

st.title("Liver Disease Predictive Framework")
st.write('This framework was created to predict liver disease in patients by Okeke')

Age = st.sidebar.slider('Age', 
                        min_value=1.00, 
                        max_value=150.00, 
                        value=30.00, 
                        step = 1.00)

Gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])

Total_Bilirubin = st.number_input('Total_Bilirubin', 
                          min_value=0.00, 
                          max_value=100.00, 
                          value=45.00)

Direct_Bilirubin = st.number_input('Direct_Bilirubin', 
                                min_value=0.00, 
                                max_value=50.00, 
                                value=15.00)

Alkaline_Phosphotase = st.number_input('Alkaline_Phosphotase', 
                                min_value=50.00, 
                                max_value=3000.00, 
                                value=100.00)

Alamine_Aminotransferase = st.number_input('Alamine_Aminotransferase', 
                          min_value=1.00, 
                          max_value=2500.00, 
                          value=100.00)

Aspartate_Aminotransferase = st.number_input('Aspartate_Aminotransferase', 
                          min_value=1.00, 
                          max_value=5000.00, 
                          value=500.00)

Total_Protiens = st.sidebar.slider('Total_Protiens', 
                                min_value=0.10, 
                                max_value=10.00, 
                                value=6.00, 
                                step = 0.10)

Albumin = st.sidebar.slider('Albumin', 
                            min_value=0.50, 
                            max_value=10.00, 
                            value=2.00, 
                            step = 0.10)

Albumin_and_Globulin_Ratio = st.sidebar.slider('Albumin_and_Globulin_Ratio', 
                            min_value=0.10, 
                            max_value=5.00, 
                            value=1.50, 
                            step = 0.10)

features = {'Age': Age, 'Gender': Gender, 'Total_Bilirubin': Total_Bilirubin, 'Direct_Bilirubin': Direct_Bilirubin,
            'Alkaline_Phosphotase': Alkaline_Phosphotase, 'Alamine_Aminotransferase': Alamine_Aminotransferase,
            'Aspartate_Aminotransferase': Aspartate_Aminotransferase, 'Total_Protiens': Total_Protiens,
            'Albumin': Albumin, 'Albumin_and_Globulin_Ratio': Albumin_and_Globulin_Ratio
            }
 

features_df  = pd.DataFrame([features])

st.table(features_df)  

if st.button('Predict'):
    
    prediction = predict_quality(model, features_df)
    
    st.write('Based on values provided, the patient has '+ str(prediction))
