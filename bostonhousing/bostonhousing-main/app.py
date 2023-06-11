import pickle
import sklearn
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models
house_model = pickle.load(open('housepredict_model.sav', 'rb'))

# page title
st.title('House Price Predictive Framework')
    
    
# getting the input data from the user
col1, col2, col3 = st.columns(3)
    
with col1:
  CRIM = st.text_input('Per Capita Crime Rate')
        
with col2:
  ZN = st.text_input('Land Zone')
    
with col3:
  INDUS = st.text_input('Non-Retail Business Acres')
    
with col1:
  CHAS = st.text_input('Charles River Variable')
    
with col2:
  NOX = st.text_input('Nitric Oxide Concentration')
    
with col3:
  RM = st.text_input('Average number of Rooms')
    
with col1:
  AGE = st.text_input('Proportion of owner-occupied units')

with col2:
  DIS = st.text_input('Boston Employment Centres')

with col3:
  RAD = st.text_input(' Index Accessibility to Highways')

with col1:
  TAX = st.text_input('Property Tax Rate')

with col2:
  PTRATIO = st.text_input('Student Teacher Ration')

with col3:
  B = st.text_input('Prop African American descent')

with col1:
  LSTAT = st.text_input('Percentage of Lower Status of the Population')

with col2:
  DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

# creating a button for Prediction
if st.button("Predict"):
    output = house_model.predict([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    output = '$' + str(output)
    st.success('The house price estimate is: {}'.format(output))
