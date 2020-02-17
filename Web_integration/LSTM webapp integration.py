#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras
import pandas as pd
import seaborn as sns 
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from dateutil.parser import parse
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import math
import datetime as dt
from keras.models import load_model as load

def lstm(df ,data, months, var):
#     if data=='iig_maitri' or data=='iig_bharati':
#         df=pd.read_csv('datasets/'+data+'.csv', parse_dates=['obstime'], index_col='obstime')
#         # print(df)
#     elif data=='dcwis':
#         df=pd.read_csv('datasets/'+data+'.csv', names=['obstime', 'tempr', 'ap', 'ws', 'rh', 'dew'], parse_dates=['obstime'], index_col='obstime')
    data2['obstime']=pd.to_datetime(data2['obstime'])
    data2=data2.set_index('obstime')
    
    ds_temp=df[var].resample('M').mean()
    if var=='rh':
        dsrh=df['rh']
        dsrh=dsrh[dsrh>10]
        # plt.plot(dsrh)
        ds_temp=dsrh.resample('M').mean()
    if var=='ws':
        dsrh=df['ws']
        dsrh=dsrh[dsrh>=0]
        # plt.plot(dsrh)
        ds_temp=dsrh.resample('M').mean()
    mean=ds_temp.mean()
    if var=='tempr':
        mean=0
    if var=='ap':
        dsrh=df['ap']
        dsrh=dsrh[dsrh>-10]
        # plt.plot(dsrh)
        ds_temp=dsrh.resample('M').mean()
    ds_temp=ds_temp-mean
    # df.tempr.plot()
    ds_temp2=ds_temp.dropna().values
    ds_temp3=ds_temp.values
    # print(ds_temp)
    # print(ds_temp2)
    
    
    
    # model=Sequential()
    # model.add(LSTM(300,input_shape=(12, 1),return_sequences=True))
    # model.add(LSTM(300))
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_error',optimizer='adam')
    ds_temp2=np.reshape(ds_temp2, (ds_temp2.shape[0], 1, 1))
    
    
    def create_dataset(dataset, look_back=1):
        dataX, dataY=[],[]
        for i in range(len(dataset)):
            end_ix=i+look_back
            if end_ix>len(dataset)-1:
                break
            seqx, seqy= dataset[i:end_ix],dataset[end_ix]
            dataX.append(seqx)
            dataY.append(seqy)
        return np.array(dataX),np.array(dataY)
    # look_back=1
    trainX, trainY=create_dataset(ds_temp2, 12)
    # print(trainX.shape)
    X=np.array(trainX).reshape(int(trainX.shape[0]), 12, 1)
    # print(X.shape)
    
    Y=np.array(trainY)
    Y=np.reshape(Y,(Y.shape[0],1))
    # print(Y.shape)
    
    
    # model.fit(X, Y, epochs=20, verbose=1, batch_size=1)
    
    
    model=load('keras_models/'+data+'_lstm_'+var+'.h5')
    # print(ds_temp2.size)
    temp_values=ds_temp2
    preds=ds_temp3
    # testX=ds_temp[29598:]
    # for i in range(60):
    #     x_input=temp_values[-24*10:]
    #     x_input=x_input.reshape(10,24,1)
    #     yhat=model.predict(x_input, verbose=1)
    #     temp_values=np.append(temp_values,yhat)
    # months=24
    for i in range(months):
        x_input=temp_values[-12:]
        x_input=x_input.reshape(1,12,1)
        # print(i,' out of ',1000,':  ')
        yhat=model.predict(x_input, verbose=0)
        temp_values=np.append(temp_values,yhat[0])
        preds=np.append(preds,yhat[0])
    
    
    dates=[]
    for year in range(2012, 2016+int(months/6)):
        for month in range (1, 13):
            dates.append(dt.datetime(year=year, month=month, day=28))
    # preds, conf_int = pipe.predict(n_periods=months, return_conf_int=True)
    # plt.subplot(211)
    plt.plot(dates[:preds.size],preds+mean)
    # plt.subplot(212)
    plt.plot(dates[:ds_temp3.size],ds_temp3+mean)
    # plt.xlim(38000,39000)
    plt.xlim(str(2012+int(preds.size/12)-6)+'-11', str(2012+int(preds.size/12)+1)+'-2')
        


# In[ ]:




