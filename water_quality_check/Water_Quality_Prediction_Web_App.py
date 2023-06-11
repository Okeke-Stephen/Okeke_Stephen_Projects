#Creating the web app with streamlit
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


def predict_quality(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    
    return predictions_data['Label'][0]
    
model = load_model('Trained_water_quality_model')


st.title('Water Quality Estimation App')
st.write('This is a web app to classify the quality of water based on\
         several features shown on the sidebar. Please adjust the\
         value of each feature. Then, click on the Predict button at the bottom to\
         see the prediction of the classifier.\
         Note: this work is for academic purposes by Okeke')


aluminium = st.sidebar.slider(label = 'Auminium', min_value = 0.00,
                          max_value = 5.5,
                          value = 2.5,
                          step = 0.1)

ammonia = st.sidebar.slider(label = 'Ammonia', min_value = 0.00,
                          max_value = 30.0,
                          value = 15.00,
                          step = 0.01)
                          
arsenic = st.sidebar.slider(label = 'Arsenic ', min_value = 0.00,
                          max_value = 1.10,
                          value = 0.50,
                          step = 0.01)  
 
barium = st.sidebar.slider(label = 'Barium', min_value = 0.0,
                          max_value = 5.0,
                          value = 2.0,
                          step = 0.2)                       

cadmium = st.sidebar.slider(label = 'Cadmium', min_value = 0.0,
                          max_value = 0.2 ,
                          value = 0.0001,
                          step = 0.001)

chloramine = st.sidebar.slider(label = 'Chloramine', min_value = 0.0,
                          max_value = 8.7,
                          value = 1.0,
                          step = 0.1)

chromium = st.sidebar.slider(label = 'Chromium', min_value = 0.000,
                          max_value = 1.000 ,
                          value = 0.500,
                          step = 0.001)
   
copper = st.sidebar.slider(label = 'Copper', min_value = 0.0,
                          max_value = 2.0,
                          value = 0.5,
                          step = 0.02)

flouride = st.sidebar.slider(label = 'Flouride', min_value = 0.0,
                          max_value = 1.5,
                          value = 0.5,
                          step = 0.001)

bacteria = st.sidebar.slider(label = 'Bacteria', min_value = 0.000,
                          max_value = 1.0,
                          value = 0.500,
                          step = 0.001)

viruses = st.sidebar.slider(label = 'Viruses', min_value = 0.000,
                          max_value = 1.0,
                          value = 0.500,
                          step = 0.001)

lead = st.sidebar.slider(label = 'Lead', min_value = 0.00,
                          max_value = 0.2,
                          value = 0.002,
                          step = 0.0002)

nitrates = st.sidebar.slider(label = 'Nitrates', min_value = 0.00,
                          max_value = 20.00 ,
                          value = 10.00,
                          step = 0.5)
                          
nitrites = st.sidebar.slider(label = 'Nitrites', min_value = 0.00,
                          max_value = 3.00,
                          value = 0.50,
                          step = 0.03)

mercury = st.sidebar.slider(label = 'Mercury', min_value = 0.0,
                          max_value = 0.01,
                          value = 0.001,
                          step = 0.001)

perchlorate = st.sidebar.slider(label = 'Perchlorate', min_value = 0.00,
                          max_value = 60.00 ,
                          value = 20.00,
                          step = 5.00)

radium = st.sidebar.slider(label = 'Radium', min_value = 0.00,
                          max_value = 8.00 ,
                          value = 2.00,
                          step = 1.00)

selenium = st.sidebar.slider(label = 'Selenium', min_value = 0.000,
                          max_value = 0.1,
                          value = 0.0500,
                          step = 0.001)

silver = st.sidebar.slider(label = 'Silver', min_value = 0.000,
                          max_value = 0.5,
                          value = 0.0500,
                          step = 0.001)


uranium = st.sidebar.slider(label = 'Uranium', min_value = 0.000,
                          max_value = 0.09,
                          value = 0.000,
                          step = 0.001)

features = {'aluminium': aluminium, 'ammonia': ammonia,
            'arsenic': arsenic, 'barium': barium,
            'cadmium': cadmium, 'chloramine': chloramine,
            'chromium': chromium, 'copper': copper,
            'flouride': flouride, 'bacteria': bacteria, 'viruses': viruses,
            'lead': lead, 'nitrates': nitrates, 'nitrites': nitrites, 'mercury': mercury,
            'perchlorate': perchlorate, 'radium': radium, 'selenium': selenium, 'silver': silver, 'uranium': uranium
            }
 

features_df  = pd.DataFrame([features])

#st.table(features_df)  

if st.button('Predict'):
    
    prediction = predict_quality(model, features_df)
    
    st.write('Based on values provided, the water quality is '+ str(prediction))
