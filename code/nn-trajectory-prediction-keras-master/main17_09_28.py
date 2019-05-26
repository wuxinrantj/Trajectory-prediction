#training script for neural nets
import numpy as np
from utils import DataLoader
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop


def shift(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = 0
        e[n:] = xs[:-n]
    else:
        e[n:] = 0
        e[:n] = xs[-n:]
    return e


seqLength = 20
obsLength = 8
predLength = 12
data_loader = DataLoader(seq_length=seqLength)  #, forcePreProcess=True)

trajectoriesTrain, trajectoriesTest = data_loader.get_data(test_dataset=4)
xA = np.array(trajectoriesTrain)
print('training set shape', xA.shape)

X = xA[:, 0:obsLength, :]
y = xA[:, obsLength, :]
model = Sequential()
model.add(Dense(64, input_shape=(obsLength, 2)))
model.add(Activation('relu'))
model.add(LSTM(128))
model.add(Dense(2, activation='linear'))

optimizer = RMSprop(lr=0.003)
model.compile(loss=losses.mean_squared_error, optimizer=optimizer)

model.fit(X, y, batch_size=128, epochs=100, verbose=2)

Xt = np.array(trajectoriesTest)
print('test set shape', Xt.shape)

squared_error = 0
final_displacement_error = 0
for Xtt in Xt:
    #print(Xtt)
    test = np.array(Xtt[0:obsLength,:])
    for i in range(obsLength,seqLength):
        pred = model.predict(np.array([test]), verbose=0)
        #print('truth', Xtt[i], 'pred', pred[0])
        squared_error += np.sum((Xtt[i] - pred[0]) ** 2)
        test = shift(test, -1)
        test[obsLength-1] = pred[0]
    final_displacement_error += np.linalg.norm(Xtt[i] - pred[0])

squared_error /= (Xt.shape[0] * predLength)
final_displacement_error /= Xt.shape[0]
print('mean square error', squared_error)
print('final displacement error', final_displacement_error)