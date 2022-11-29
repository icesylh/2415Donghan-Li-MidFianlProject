import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#lstmModels
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from matplotlib.pyplot import MultipleLocator
import matplotlib.dates as mdate

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#New construction dataframe，Months as a unit
years = [2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
preyears = [21,22]
months = ['5', '6', '7', '8', '9', '10', '11', '12', '1', '2', '3', '4']
num = 60  #60-80 30-50

#Get monthly average
def pinjun(data):
    #Create a new dataframe
    rebuild = []
    for row in data.itertuples():
        cu_day = getattr(row, 'Date')  # Output each line
        cu_day = str (cu_day)
        a = datetime.strptime(cu_day, '%Y-%m-%d %H:%M:%S').strftime('%y-%m')
        rebuild.append(a)
    data.insert(data.shape[1], 'mon', rebuild)
    data1 = data[["mon", "Open", "Close"]]
    mondata = data1.groupby('mon').mean()
    return mondata

#Models
def lstm(df,mondata,item):
    # Create Data Box
    # # mondata = mondata.reset_index()
    # data = df.sort_index(ascending=True, axis=0)
    # new_data = pd.DataFrame(index=range(0, len(data)), columns=['Date', item,'mon'])
    # for i in range(0, len(data)):
    #     new_data.loc['Date'][i] = data['Date'][i]
    #     new_data.loc[item][i] = data[item][i]
    # # Set Index
    # new_data.index = new_data.Date
    # new_data.drop('Date', axis=1, inplace=True)
    # print(new_data)

    new_data = df[[item]]
    # new_data = mondata[[item]]

    # Create training set and validation set
    dataset = new_data.values

    # train = dataset[0:100, :]  #That's a total of 120 data
    # valid = dataset[100:, :]
    train = dataset[0:30000, :]
    valid = dataset[30000:, :]

    # 将数据集转换为x_train和y_train
    scaler = MinMaxScaler(feature_range=(0, 1))  #Data normalization process
    scaled_data = scaler.fit_transform(dataset)  #Standardization

    x_train, y_train = [], []
    for i in range(num,len(train)):  #Use num pieces to predict
        x_train.append(scaled_data[i - num:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Creating and fitting LSTM networks
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    ave_data = mondata[[item]]
    inputs = new_data[len(new_data) - len(valid) - num:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    # for i in range(inputs.shape[0]):     #.shape[0]The number of rows of the array is added here with 24
    #     X_test.append(inputs[:, 0])    #Dimension1
    for i in range(num, inputs.shape[0]):
        X_test.append(inputs[i - num:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    print('-------------')
    print(len(closing_price))  #ndarray
    print('-------------')
    closing_price1 = closing_price[0:4554:100]

    # 计算rms
    # rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))
    # rms = np.sqrt(np.mean(np.power((valid - closing_price[:20]), 2)))    #The first 20 are used to verify
    # print('---------------------')
    # print(rms)
    # print('---------------------')

    # For drawing
    train = ave_data[:100] #Top 100
    valid = ave_data[100:]
    # valid['Predictions'] = closing_price[:20]
    valid.insert(valid.shape[1], 'Predictions', closing_price[:20])  #Top20
    #Create predictive data indexes
    preindex = []
    for y in preyears:
        for m in months:
            if m in ['5', '6', '7', '8', '9', '10', '11', '12']:
                date = str(y) + "-" + str(m)
            else:
                date = str(y+1) + "-" + str(m)
            preindex.append(date)
    # Predictive data dataframe
    predict = pd.DataFrame(closing_price1[len(closing_price1)-24:], index = preindex, columns=[ item ])  #End24


    plt.plot(train[ item ],label = "train", color='b')
    plt.plot(valid[ item ],label = "test", color='y')
    plt.plot(valid['Predictions'],label = "testpre", color='g')
    plt.plot(predict[ item ],label = "predict", color='r')
    plt.legend(loc='upper left')
    ax = plt.gca()
    x_major_locator = MultipleLocator(12)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xticks(rotation=30)
    plt.title('Average '+item+' Price of Each Month', fontsize=14)
    plt.xlabel('time', fontsize=14)
    plt.ylabel('price', fontsize=14)
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv('./indexData.csv')  #Data Reading
    #Data pre-processing
    data['Date'] = pd.to_datetime(data['Date'])  # Convert the time in csv
    data.sort_values('Date', inplace=True)  # Disordered time rearrangement
    data.dropna(axis=0, how='any', inplace=True, subset=None)  # Remove rows with null values

    # Selected data for 12 months per year for the decade 2011-2021
    open_day = pd.to_datetime('2011-05-01')
    close_day = pd.to_datetime('2021-04-30')
    con1 = data['Date'] >= open_day
    con2 = data['Date'] < close_day
    df = data[con1 & con2]

    # Set index to date
    df.index = df['Date']

    #open and close each month average dataframe format
    mondata= pinjun(df)

    #lstm Switch opening or closing prices according to demand
    item = 'Close'
    # item = 'Open'
    lstm(df,mondata,item)





