#training script for neural nets
import numpy as np
from utils import DataLoader
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
batchSize = 50
seqLength = 20
obsLength = 8
predLength = 12
selectedData = [0,1,2,3,4]
data_loader = DataLoader(batchSize, seqLength, selectedData)

trajectories = data_loader.get_data()
xA = np.asarray(trajectories)

X = xA[:,0:obsLength,:]
y = xA[:,obsLength,:]
model = Sequential()
model.add(Dense(64, input_shape=(obsLength, 2)))
model.add(Activation('relu'))
#model.add(LSTM(128, input_shape=(obsLength, 2)))
model.add(LSTM(128))
model.add(Dense(2))
model.add(Activation('linear'))

optimizer = RMSprop(lr=0.003)
model.compile(loss=losses.mean_squared_error, optimizer=optimizer)

personIndex = 4
test = np.array([xA[personIndex, 0:seqLength - 1, :]])
truth = xA[personIndex, seqLength - 1, :]
print('truth', truth)

for iteration in range(1, 50):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=128,
              epochs=1)

'''pred = model.predict(test,verbose=0)
print('pred',pred)'''