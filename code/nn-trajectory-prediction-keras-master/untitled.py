import numpy as np
import tensorflow as tf
from utils import *
from tensorflow.python.ops import rnn_cell
import matplotlib.pyplot as plt

seqLength = 20
selectedData = [0,1,2,3,4]
data_loader = DataLoader(seq_length=seqLength, datasets=selectedData, forcePreProcess=True)
trajectoriesTrain, trajectoriesTest = data_loader.get_data(test_dataset=4)
xA = np.array(trajectoriesTest)

print(xA.shape)



'''X = np.array([[0., 0., 1., 1., 0.5], [0., 0., 2., 2., 0.5],
             [0., 0., 3., 3., 0.5], [0., 0., 4., 4., 0.5]], dtype='float32')
y = np.array([[0.5, 0.6, 0, 0, 0], [1.5, 1.6, 0, 0, 0],
             [2.5, 2.6, 0, 0, 0], [3.5, 3.6, 0, 0, 0]], dtype='float32')
X2 = np.array([[[0., 0., 3., 3., 0.5], [0., 0., 4., 4., 0.5]],
               [[0., 0., 1., 1., 0.5], [0., 0., 2., 2., 0.5]]], dtype='float32')

#print(log_likelihood_error(y, X))
sess = tf.Session()
nll = negative_log_likelihood(y, X)
nllAuthor = categorical_hinge(y, X)
print('nlls', sess.run(nllAuthor), sess.run(nll))'''

