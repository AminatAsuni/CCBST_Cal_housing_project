import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

st.title("California Housing Price Prediction for XYZ Brokerage Company")
st.write("Hello world please use cal house price prediction")

# load the dataset
cal = fetch_california_housing()
df=pd.DataFrame(cal.data,columns=cal.feature_names)
df['MedHouseVal']=cal.target

X=df.drop('MedHouseVal', axis =1) #Input Features
y=df['MedHouseVal'] #target Feature

st.subheader("Data Overview for the first 10 rows")
st.dataframe(df.head(10))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scaling the data
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# create a linear regression model
model = LinearRegression() # initialize the model
model.fit(X_train_sc, y_train) #fit the model

# make predictions
y_pred = model.predict(X_test_sc)

# evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance Metrics")
st.write(f"Mean Squared Error: {mse}")
st.write(f"Mean Absolute Error: {mae}")
st.write(f"R-squared: {r2}")

#get user input
st.write ("Enter the input values for prediction:")

user_input={}

for col in X.columns:
    user_input[col] = st.number_input(col)

#convert user input to datframe
user_input_df = pd.DataFrame(user_input,index=[0])

#scale the user input
user_input_sc = scaler.transform(user_input_df)

#make prediction
prediction = model.predict(user_input_sc)

st.button("Predict")

st.write(f"Predicted House Price:{np.round(prediction[0]*100000)}")