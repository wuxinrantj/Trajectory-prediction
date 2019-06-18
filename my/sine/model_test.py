from keras.models import load_model
from matplotlib import pyplot
from generate import *
import numpy as np

model = load_model('MAE:0.131105.h5')
length = 200
output = 20

X, y, p, d = generate_examples(length, 1, output)
yhat = model.predict(X, verbose=0)


y = generate_sequence(length + output, p, d)
pyplot.figure()
pyplot.plot(y, label='y')
pyplot.plot(range(200,220,1), yhat[0], label='yhat')
pyplot.title("MSE: {:f}".format(0.0131105))
pyplot.legend()
pyplot.show()

y_truth = []
y_predict = []
error = []
pyplot.figure()
for i in range(20):
    y_truth.append(5 - np.sqrt(10-y[i+200]*y[i+200]))
    y_predict.append(5 - np.sqrt(10-yhat[0][i]*yhat[0][i]))
    error.append(y_truth[i] - y_predict[i])
pyplot.scatter(y[200:], y_truth, marker='o', c='', edgecolors='g', label='truth')
pyplot.scatter(y[200:], y_predict, marker='*', c='r', edgecolors='r', label='predict')
pyplot.legend()
pyplot.show()

pyplot.figure()
pyplot.plot(range(20), error, label='error')
pyplot.legend()
pyplot.show()
