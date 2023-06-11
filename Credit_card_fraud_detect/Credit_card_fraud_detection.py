#Creating the web app with streamlit
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


def predict_quality(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    
    return predictions_data['Label'][0]
    
model = load_model('credit_card_dt_deployment_model')


st.title('Credit Card Fraud Detection framework')
st.write('This is a web-based framework to detect credit card fraud in transactions by Okeke')


Distance_from_home = st.sidebar.slider(label = 'distance_from_home', min_value = 0.001,
                          max_value = 10632.734,
                          value = 5000.000,
                          step = 10.000)

Distance_from_last_transaction = st.sidebar.slider(label = 'distance_from_last_transaction', min_value = 0.0001,
                          max_value = 11851.105,
                          value = 100.000,
                          step = 5.000)
                          
Ratio_to_median_purchase_price = st.sidebar.slider(label = 'ratio_to_median_purchase_price', min_value = 0.005,
                          max_value = 267.803,
                          value = 50.000,
                          step = 5.000)  
 
Repeat_retailer = st.sidebar.slider(label = 'repeat_retailer', min_value = 0.000,
                          max_value = 1.000,
                          value = 0.500,
                          step = 0.100)                       

Used_chip = st.sidebar.slider(label = 'used_chip', min_value = 0.000,
                          max_value = 1.000,
                          value = 0.500,
                          step = 0.100)

Used_pin_number = st.sidebar.slider(label = 'used_pin_number', min_value = 0.000,
                          max_value = 1.000,
                          value = 0.500,
                          step = 0.100)

Online_order = st.sidebar.slider(label = 'online_order', min_value = 0.000,
                          max_value = 1.000,
                          value = 0.500,
                          step = 0.100)

features = {'Distance_from_home': Distance_from_home, 'Distance_from_last_transaction': Distance_from_last_transaction,
            'Ratio_to_median_purchase_price': Ratio_to_median_purchase_price, 'Repeat_retailer': Repeat_retailer,
            'Used_chip': Used_chip, 'Used_pin_number': Used_pin_number,
            'Online_order':Online_order}
 

features_df  = pd.DataFrame([features])

st.table(features_df)  

if st.button('Predict'):
    
    prediction = predict_quality(model, features_df)
    
    st.write('Based on the parameters provided, the transaction is {}'+ str(prediction))