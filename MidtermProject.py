import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#Draw bar graphs
def plotbar(max):
    fig = plt.figure(figsize=(12, 5))
    x = ['2020-1','2020-2','2020-3','2020-4','2020-5','2020-6',
         '2020-7','2020-8','2020-9','2020-10','2020-11','2020-12',
         '2021-1','2021-2','2021-3','2020-4','2020-5','2021-6']
    plt.bar(x, max, 0.4, color="red")
    plt.xlabel("Months")
    plt.ylabel("Max Close in each months")
    plt.title("High Price chart")
    plt.savefig("barChart.png")
    plt.xticks(rotation=45)
    plt.show()


#Get the maximum value for the month
def monmax(data):  #[['Date','Close]]
    max = []
    y2020 = data[data['Date'].dt.year.isin(np.arange(2020,2021))]

    for i in range(1, 12):
        mondata = y2020[y2020['Date'].dt.month.isin(np.arange(i,i+1))]
        max.append(mondata['Close'].max())
    open_day = '2020-12-01'
    close_day = '2020-12-31'
    con1 = y2020['Date'] >= open_day
    con2 = y2020['Date'] < close_day
    mondata = y2020[con1 & con2]
    max.append(mondata['Close'].max())

    y2021 = data[data['Date'].dt.year.isin(np.arange(2021, 2022))]
    for i in range(1, 7):
        mondata = y2021[y2021['Date'].dt.month.isin(np.arange(i, i + 1))]
        max.append(mondata['Close'].max())

    return max


def buquan(order_data):
    # Remove duplicate dates
    order_data.drop_duplicates(inplace=True)
    order_data = order_data.reset_index(drop=True)  # Reset Index
    # Complementary vacancy date, mean value complements other values
    order_data = order_data.set_index(pd.to_datetime(order_data['Date'])).drop('Date', axis=1)
    order_data = order_data.resample('D').mean().interpolate()

    return order_data


if __name__ == "__main__":
    data = pd.read_csv('./indexData.csv')  #Data Reading
    data['Date'] = pd.to_datetime(data['Date'])  #Convert the time in csv

    data.dropna(axis=0, how='any', inplace=True, subset=None)  #Remove rows with null values

    data.drop_duplicates(subset=['Date'], keep='first', inplace=True)  #Only one date duplicate is retained
    data = data.reset_index(drop=True)  # Reset Index

    data.sort_values('Date', inplace=True)  #Disordered time rearrangement
#
    # Select data for the 2019-2021 time period
    open_day = '2018-12-30'
    close_day = '2021-12-31'
    con1 = data['Date'] >= open_day
    con2 = data['Date'] < close_day
    order_data = data[con1 & con2]


    # # Histogram data processing
    # # Select data for the time period 2020.6-2021.6
    open_day = '2019-06-01'
    close_day = '2021-06-30'
    con1 = order_data['Date'] >= open_day
    con2 = order_data['Date'] < close_day
    datanew = order_data[con1 & con2]
    bar_data = datanew[['Date','Close']]

    max = monmax(bar_data)    #Sub-function to get the maximum value in a month
    plotbar(max)  #Sub-functions to draw histograms
