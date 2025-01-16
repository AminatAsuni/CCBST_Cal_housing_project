import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("California Housing Price Prediction for XYZ Brokerage Company")
st.write("Hello world please use cal house price prediction")

#reading linear regression model through pickle file
model_linear = pickle.load(open('model_linear_reg.pkl','rb'))

#reading linear regression model through pickle file
model_scaler = pickle.load(open('scaler_cal.pkl','rb'))

#get user input
st.write ("Enter the input values for prediction:")

user_input={}

user_input['MedInc'] =st.number_input('MedInc', value =0.0)
user_input['HouseAge'] =st.number_input('HouseAge', value =0.0)
user_input['AveRooms'] =st.number_input('AveRooms', value =0.0)
user_input['AveBedrms'] =st.number_input('AveBedrms', value =0.0)
user_input['Population'] =st.number_input('Population', value =0.0)
user_input['AveOccup'] =st.number_input('AveOccup', value =0.0)
user_input['Latitude'] =st.number_input('Latitude', value =0.0)
user_input['Longitude'] =st.number_input('Longitude', value =0.0)


#convert user input to datframe
user_input_df = pd.DataFrame(user_input,index=[0])

#scale the user input
user_input_sc = model_scaler.transform(user_input_df)

#make prediction
prediction = model_linear.predict(user_input_sc)

st.button("Predict")

st.write(f"Predicted House Price:{np.round(prediction[0]*100000)}")