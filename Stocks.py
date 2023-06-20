#!pip install pandas_datareader
#!pip install --upgrade pandas_datareader

#!pip install --upgrade pandas_datareader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
import tensorflow as tf


start = '2010-01-01'
end = '2020-12-31'
st.title('Stock Trend Prediction')
user_input=st.text_input('Enter Stock Ticker', 'AAPL')

data = yf.download(user_input, start=start, end=end)

#describing data
st.subheader('Data from 2010 - 2020')
st.write(data.describe())

#VISUALIZATION

st.subheader('Closing Price Vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(data.Close)
st.pyplot(fig) #chart of closing price

st.subheader('Closing Price vs Time chart with 100 MA')
ma100 = data.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(data.Close)
st.pyplot(fig) #chart of closing price

st.subheader('Closing Price vs Time chart with 100MA & 200 MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(data.Close, 'b')
st.pyplot(fig) #chart of closing price

#splitting data into training and testing

data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)



#Load my Model 
model=load_model('keras_model.h5')

#Testing part
past_100_days=data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data=scaler.fit_transform(final_df)


x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

#making predictions
y_predicted=model.predict(x_test)

scaler=scaler.scale_

scale_factor=1/(scaler[0])
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#Final Chart
st.subheader('Predictions Vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)