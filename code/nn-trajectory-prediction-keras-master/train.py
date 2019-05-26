import numpy as np
from utils import DataLoader
from utils import negative_log_likelihood_sequences, negative_log_likelihood, mean_squared_error, \
    negative_log_likelihood_w, shift, sample_gaussian_2d
from keras import losses
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM, RepeatVector
from keras.optimizers import RMSprop


def train_many2one(X, y):
    print('observation', X.shape, 'prediction', y.shape)
    model = Sequential()
    model.add(Dense(64, input_shape=(obsLength, 2), use_bias=True, activation='relu'))
    model.add(LSTM(units=128, use_bias=False, return_sequences=False))
    model.add(Dense(5, use_bias=False, activation='linear'))
    model.summary()

    optimizer = RMSprop(lr=0.003, decay=0.95)

    model.compile(loss=negative_log_likelihood, optimizer=optimizer)
    yPadded = np.pad(y, ((0, 0), (0, 3)), 'constant', constant_values=(0., 0.))
    print('yPadded', yPadded.shape)
    history = model.fit(X, yPadded, batch_size=50, epochs=100, verbose=2)
    # model.save('many_to_one.h5')
    return history


def train_many2many(X, y):
    print('observation', X.shape, 'prediction', y.shape)
    model = Sequential()
    model.add(Dense(64, input_shape=(obsLength, 2), use_bias=True, activation='relu'))
    model.add(LSTM(units=128, use_bias=False, return_sequences=False))
    model.add(RepeatVector(predLength))
    model.add(LSTM(units=128, use_bias=False, return_sequences=True))
    model.add(TimeDistributed(Dense(5, use_bias=False, activation='linear')))
    model.summary()

    optimizer = RMSprop(lr=0.003)

    model.compile(loss=negative_log_likelihood_sequences, optimizer=optimizer)
    yPadded = np.pad(y, ((0, 0), (0, 0), (0, 3)), 'constant', constant_values=(0., 0.))  # the values is not important
    print('yPadded', yPadded.shape)
    history = model.fit(X, yPadded, batch_size=50, epochs=1, verbose=2)
    # model.save('many_to_many.h5')
    return history


seqLength = 20
obsLength = 8
predLength = 12
data_loader = DataLoader(seq_length=seqLength)  # , forcePreProcess=True)

trajectoriesTrain, trajectoriesTest = data_loader.get_data(test_dataset=4)
xA = np.array(trajectoriesTrain, dtype='float32')
print('training set shape', xA.shape)

X = xA[:, 0:obsLength, :]
y = xA[:, obsLength:seqLength, :]
xOneFrame = xA[:, 0, :]
yOneFrame = xA[:, obsLength, :]

hist = train_many2one(X, yOneFrame)
# hist = train_many2many(X, y)


import matplotlib.pyplot as plt
#print(hist.history['loss'])
plt.plot(hist.history['loss'], marker='x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()