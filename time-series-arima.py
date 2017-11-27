import json
import datetime
import time
import pandas as pd
from matplotlib import pyplot
from pandas import TimeGrouper
import matplotlib.pylab as plt

json_file = open('DC_2_Dataset_support_requests.json')

for m in json_file:
    time_stamp = json.loads(m)


arr = time_stamp['time']
time_list = map(str,arr)

dates = [datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in time_list]

#Hopefully O(NlogN) complexity
dates.sort()
sorteddates = [datetime.datetime.strftime(ts, "%Y-%m-%d %H:%M:%S") for ts in dates]

maximum_difference = 15
k = 0
count = 0
pointer_i = 0
pointer_j = 0
print sorteddates[0:35]
frequency = []
star_time_stamp = []
print len(sorteddates)
#O(N) time complexity
flag = 0
while pointer_j < len(sorteddates):
    if flag == 0:
        d1 = datetime.datetime.strptime(sorteddates[pointer_i], "%Y-%m-%d %H:%M:%S")
    else:
        d1 = next_time
    d2 = datetime.datetime.strptime(sorteddates[pointer_j], "%Y-%m-%d %H:%M:%S")
    d1_ts = time.mktime(d1.timetuple())
    d2_ts = time.mktime(d2.timetuple())
    if int(d2_ts - d1_ts) / 60 <= 15:
        count = count + 1
        pointer_j = pointer_j + 1

    else:
        next_time = d1 + datetime.timedelta(minutes=15)
        #print sorteddates[pointer_j - 1]
        #print sorteddates[pointer_j]
        start_date = d1.strftime('%Y-%m-%d %H:%M:%S')
        star_time_stamp.append(start_date)
        frequency.append(count)
        count = 0
        flag = 1

    if pointer_j == len(sorteddates):
        start_date = d1.strftime('%Y-%m-%d %H:%M:%S')
        star_time_stamp.append(start_date)
        frequency.append(count)
        print 'it is completed'


time_series_df = pd.DataFrame(
    {'TimeStamp': star_time_stamp,
     'Demand': frequency,
    })


series = pd.Series(data= list(time_series_df['Demand']), index=pd.DatetimeIndex(time_series_df['TimeStamp']))
print series.head()
from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=96)
    rolstd = pd.rolling_std(timeseries, window=96)

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput

test_stationarity(series[16:])

'''Here one can see there is no constant mean nor constant  aand also test statistic is greater than the critical value'''
import numpy as np
'''So one need to eliminate the TS'''
series = series[16:]
print series[192:288]
differenced = series.diff(480)


# trim off the first week of empty data
differenced = differenced[480:]
# save differenced dataset to file
#differenced.to_csv('seasonally_adjusted.csv')
test_stationarity((differenced))

'''This shows the time series is stationary where Test statistic is less than the critical value '''
#moving_avg = pd.rolling_mean(ts_log,96)
#plt.plot(ts_log)
#plt.plot(moving_avg, color='red')
#plt.show()

ts_log_diff = differenced
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=50)
lag_pacf = pacf(ts_log_diff, nlags=50, method='ols')
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#plt.show()
#Shows q = 28


plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
#plt.show()
#Shows p = 5

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log_diff, order=(5, 0,2))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
plt.show()


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print predictions_ARIMA_diff.head()