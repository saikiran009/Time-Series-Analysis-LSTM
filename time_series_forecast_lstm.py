import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import json
import datetime
import time
import pandas as pd
from matplotlib import pyplot
from pandas import TimeGrouper
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from numpy import newaxis
import itertools




'''Load the data'''


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

'''REstructure the given timeseries into uniform 15-minute window based'''
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

'''Create a pandas dataframe '''
time_series_df = pd.DataFrame(
    {'TimeStamp': star_time_stamp,
     'Demand': frequency,
    })


'''Convert the given dataset into numpy array with 3 -inputs'''
def load_data(data, seq_len, normalise_window):

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    print train
    np.random.shuffle(train) #'''Shuffle randomly the training data set'''
    x_train = train[:, :-1]
    print x_train.shape
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]

denormalise_multi_data = []
'''Normalise the data for RNN's to train more quickly'''
def normalise_windows(window_data):
    normalised_data = []

    #print window_data
    #print 'these is the window'
    for window in window_data:
        #print window
        #print 'thisis shit '
        summation = sum(window)
        normalised_window = [(float(p) / float(summation)) for p in window]
        normalised_data.append(normalised_window)
        denormalise_data = [summation] * len(normalised_window)
        denormalise_multi_data.append(denormalise_data)
        #print normalised_data
        #print 'this is the shit'
    return normalised_data

'''Build two LSTM models'''
def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    model.compile(loss="mse", optimizer="rmsprop")
    return model





X_train, y_train, X_test, y_test = load_data(frequency, 96, True)
print len(X_train)
print len(y_train)
print len(X_test)
print len(y_test)
model = build_model([1, 96, 100, 1])
'''Fit the model'''
model.fit(X_train,y_train,batch_size= 512,nb_epoch= 30,validation_split=0.05)

'''Predict the model'''
def predict_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

#predictions = predict_sequences_multiple(model, X_test, 96, 4)
predicted = predict_point(model, X_test)
#print len(predicted)
#print denormalise_multi_data
flat_denormalise = list(itertools.chain(*denormalise_multi_data))

'''Denormalise the data'''
def bring_back(flat_denormalise,predicted,y_test):
    #predicted_ = [x-1  for x in predicted]
    #print predicted_
    actual_predicted = [a * b for a, b in zip(flat_denormalise, predicted)]

    actual_real = [a * b for a, b in zip(flat_denormalise, y_test)]
    print actual_predicted[-4:]
    print actual_real[-4:]
bring_back(flat_denormalise,predicted,y_test)



'''Plot the data for visulisation'''

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

plot_results(predicted, y_test)




