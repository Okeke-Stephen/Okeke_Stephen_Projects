#Deploying the trained model with streamlit
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('Heart_attack_pre_deployment_model')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app was created to predict heart attacks by Okeke')

    st.title("Heart Attacks Predictive Framework")

    if add_selectbox == 'Online':

        #age = st.number_input('Age', min_value=1, max_value=100, value=25)
        Age = st.sidebar.slider('Age', min_value=1, max_value=100, value=15, step = 1)
        Sex = st.selectbox('Sex', ['male', 'female'])
        ChestPainType = st.text_input("ChestPainType", 'ASY')
        RestingBP = st.number_input('RestingBP', min_value=0, max_value=200, value=120)
        Cholesterol = st.number_input('Cholesterol', min_value=0, max_value=800, value=50)
        FastingBS =  st.sidebar.slider('FastingBS', min_value=0.0, max_value=1.0, value=0.5, step = 0.1)
        RestingECG = st.text_input("RestingECG", 'ST')
        MaxHR = st.number_input('MaxHR', min_value=10, max_value=250, value=50)
        ExerciseAngina = st.selectbox('ExerciseAngina', ['Y', 'N'])
        Oldpeak = st.number_input('Oldpeak', min_value=-0.0, max_value=10.0, value=3.0)
        ST_Slope = st.text_input("ST_Slope", 'Up')
       
        output=""

        input_dict = {'Age' : Age, 'Sex' : Sex, 'ChestPainType' : ChestPainType, 'RestingBP' : RestingBP, 'Cholesterol' : Cholesterol, 'FastingBS' : FastingBS, 
                      'RestingECG' : RestingECG,'MaxHR' : MaxHR,'ExerciseAngina' : ExerciseAngina,'Oldpeak' : Oldpeak, 'ST_Slope' : ST_Slope}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)

        st.success('The patient is has {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
