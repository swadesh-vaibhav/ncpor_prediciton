#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pmdarima as pm
from pmdarima import pipeline, preprocessing as ppc, arima
from matplotlib import pyplot as plt
import math
import datetime as dt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import pandas as pd
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats

def Arima(df, dataset, months, var):
#     if dataset=='iig_maitri' or dataset=='iig_bharati':
#         data2=pd.read_csv('datasets/'+dataset+'.csv')
#         # print(df)
#     elif dataset=='dcwis':
#         data2=pd.read_csv('datasets/'+dataset+'.csv', names=['obstime', 'tempr', 'ap', 'ws', 'rh', 'dew'])
    df['obstime']=pd.to_datetime(df['obstime'])
    df=df.set_index('obstime')
    data2=df[var]
    # print(data2)
    # train2, test2= data2[:23423], data2[23423:]
    # train =train2.resample('M').mean()
#     ds_temp=df[var].resample('M').mean()
    if var=='rh':
        data2=data2[data2>10]
    if var=='ws':
        data2=data2[data2>=0]
    if var=='ap':
        data2=data2[data2>-10]  
    data =data2.resample('M').mean()
    # print(data)
    # data.dropna(inplace=True)
    # train.dropna(inplace=True)
    # test.dropna(inplace=True)
    #     data['Date']=data.index.strftime('%B')
    #     data['Date']=data['Date']+' '+data.index.strftime('%Y')
    # data.set_index('Date', inplace=True)
#     datum=data
    data.dropna(inplace=True)
    # print(data.size)
    ind=list()
    for i in range (int(data.size)):
        ind.append(i)
    # print(data.Date)
    # print(ind)
#     Q1 = data.quantile(0.25)
#     Q3 = data.quantile(0.75)
#     IQR = Q3 - Q1
#     # print(IQR)
#     data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR)))]
    # data.shape
    # Let's create a pipeline with multiple stages... the Wineind dataset is
    # seasonal, so we'll include a FourierFeaturizer so we can fit it without
    # seasonality
    pipe = pipeline.Pipeline([
        ("fourier", ppc.FourierFeaturizer(m=12)),
        ("arima", arima.AutoARIMA(stepwise=True, trace=1, error_action="ignore",
                                  seasonal=False,  # because we use Fourier
                                  transparams=False,
                                  suppress_warnings=True))
    ])

    pipe.fit(data)
    # print("Model fit:")
#     # print(pipe)
#     months=12
    dates=[]
    for year in range(2012, 2016+int(months/6)):
        for month in range (1, 13):
            dates.append(dt.datetime(year=year, month=month, day=28))
    preds, conf_int = pipe.predict(n_periods=months, return_conf_int=True)
    datum=data
    data3=data2.resample('M').mean()
    # print(dates)
    # print(data3[dt()'2012-01-31'])
    # for i, j in zip(data3.index, range(data3.size)):
    #     if math.isnan(data3[i])==True:
    #        del dates[j]
    # print(datum)
    temp=np.append(data3,preds)
    # dates=np.array(dates)
    # print(np.array(dates).shape)
    # # print
    # print(temp.shape)
    # plt.subplot(211)
    plt.plot(dates[:temp.size], temp)
    plt.plot(datum)
    # print(temp)
    # plt.subplot(212)

