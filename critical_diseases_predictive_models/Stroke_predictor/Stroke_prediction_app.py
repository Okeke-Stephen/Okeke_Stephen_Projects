#Deploying the trained model with streamlit
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('stroke_deployment_model')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This framework was created to predict stroke attacks by Okeke')

    st.title("Stroke Attacks Predictive Framework")

    if add_selectbox == 'Online':

        Gender = st.selectbox('Gender', ['male', 'female', 'Other'])
        Age = st.sidebar.slider('Age', min_value=0.5, max_value=100.0, value=15.0, step = 0.5)
        Hypertension = st.selectbox('Hypertension', ['0', '1'])
        Heart_disease = st.selectbox('Heart_disease', ['0', '1'])
        Ever_married = st.text_input("Ever_married", 'No')
        Work_type = st.text_input("Work_type", 'Govt_job')
        Residence_type = st.text_input("Residence_type", 'Urban')
        Avg_glucose_level = st.number_input('Avg_glucose_level', min_value=10.00, max_value=400.00, value=70.00)
        Bmi = st.number_input('Bmi', min_value=10.00, max_value=200.00, value=30.00)
        Smoking_status = st.text_input("Smoking_status", 'Unknown')

        output=""

        input_dict = {'Gender' : Gender, 'Age' : Age, 'Hypertension' : Hypertension, 'Heart_disease' : Heart_disease, 'Ever_married' : Ever_married, 'Work_type' : Work_type, 
                      'Residence_type' : Residence_type,'Avg_glucose_level' : Avg_glucose_level,'Bmi' : Bmi,'Smoking_status' : Smoking_status}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)

        st.success('Based on the information provided, the patient has {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
