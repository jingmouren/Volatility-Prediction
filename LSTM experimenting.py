# This program is a first attempt to train an LSTM model using just the book data from stock 0

import pandas as pd
import numpy as np
import volat as vl
from tensorflow import keras

# We read and rescale the data

frame1 = vl.parquet_frame('/home/abatsis/Λήψεις/optiver-realized-volatility-prediction/book_train.parquet/stock_id=0')
frame1[frame1.columns[2:11]] -= frame1[frame1.columns[2:11]].min()
frame1[frame1.columns[2:11]] /= frame1[frame1.columns[2:11]].max()

# We create train and test random samples

patches = vl.file_sample(frame1, 5/127)
[frame1, test_frame] = vl.patches_to_frame(frame1, patches, 4/5)
frame1 = frame1.sort_values(by=['time_id', 'seconds_in_bucket'])
frame1 = pd.DataFrame.reset_index(frame1)
frame1 = frame1.drop('index', axis=1)
test_frame = test_frame.sort_values(by=['time_id', 'seconds_in_bucket'])
test_frame = pd.DataFrame.reset_index(test_frame)
test_frame = test_frame.drop('index', axis=1)
frame1.insert(0, 'stock_id', [0]*frame1.shape[0])
test_frame.insert(0, 'stock_id', [0]*test_frame.shape[0])

# We read and rescale the respective target values

values_fr = vl.csv_frame('/home/abatsis/Λήψεις/optiver-realized-volatility-prediction/train.csv')
values_fr[['target']] -= values_fr[['target']].min()
values_fr[['target']] /= values_fr[['target']].max()
m = 0
while values_fr.loc[m].at["stock_id"] == 0:
    m = m+1
values_fr = values_fr[0:m]

y = vl.frame_to_values(frame1, values_fr)
y_test = vl.frame_to_values(test_frame, values_fr)

# We create the training input for the LSTM model

inpt = vl.lstm_input(frame1, 1000, 2, 11)

# We build and train the LSTM model

model = keras.Sequential()
model.add(keras.layers.LSTM(50, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(60, return_sequences=True))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.LSTM(80, return_sequences=True))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.LSTM(120, return_sequences=False))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(inpt, y, epochs=1)

# We compare our comparisons to y_test

test_data = vl.lstm_input(test_frame, 1000, 2, 11)
prediction = model.predict(test_data)
print(prediction)
print(y_test)






