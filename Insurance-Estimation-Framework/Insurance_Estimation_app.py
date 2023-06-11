#Deploying the trained model with streamlit
from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('ins_deployment_model')
#model = load_model('ins_deployment_model')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    #image = Image.open('logo.png')
    #image_hospital = Image.open('hospital.jpg')

    #st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app was created to predict patient hospital insurance charges by Okeke')
    #st.sidebar.success('https://www.pycaret.org')
    
    #st.sidebar.image(image_hospital)

    st.title("Insurance Charges Prediction Framework")

    if add_selectbox == 'Online':

        #age = st.number_input('Age', min_value=1, max_value=100, value=25)
        age = st.sidebar.slider('Age', min_value=1, max_value=100, value=15, step = 1)
        sex = st.selectbox('Sex', ['male', 'female'])
        bmi = st.sidebar.slider('BMI', min_value=10, max_value=80, value=20, step = 1)
        #children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
        children = st.sidebar.slider('Children', min_value=0, max_value=10, value=0, step = 1)
        if st.checkbox('Smoker'):
            smoker = 'yes'
        else:
            smoker = 'no'
        region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        output=""

        input_dict = {'age' : age, 'sex' : sex, 'bmi' : bmi, 'children' : children, 'smoker' : smoker, 'region' : region}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('The insurance estimated charge is: {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
