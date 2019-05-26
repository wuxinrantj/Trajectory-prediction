import numpy as np
from utils import DataLoader
from utils import negative_log_likelihood_sequences, negative_log_likelihood, mean_squared_error, negative_log_likelihood_w, shift, sample_gaussian_2d
from keras import losses
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM, RepeatVector
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

model = load_model('many_to_one.h5',custom_objects={'negative_log_likelihood': negative_log_likelihood})

seqLength = 20
obsLength = 8
predLength = 12
datatest = 4
data_loader = DataLoader(seq_length=seqLength)  #, forcePreProcess=True)

trajectoriesTrain, trajectoriesTest = data_loader.get_data(test_dataset=datatest)

Xt = np.array(trajectoriesTest)
print('test set shape', Xt.shape)

squared_error = 0
final_displacement_error = 0

counter = 1
for Xtt in Xt:
    #print('Xtt',Xtt)
    observed = np.array(Xtt[0:obsLength,:])
    truth = np.array(Xtt[obsLength:seqLength,:])
    prediction = np.array(Xtt[obsLength:seqLength,:])
    for i in range(obsLength,seqLength):
        predicted_params = model.predict(np.array([observed]), verbose=0)
        #print('params', predicted_params.shape, predicted_params)
        pred = sample_gaussian_2d(predicted_params)
        #print('truth', Xtt[i], 'pred', pred[1], pred[0])
        prediction[i-obsLength,0] = pred[1]
        prediction[i-obsLength,1] = pred[0]
        squared_error += np.sum((Xtt[i] - pred[0]) ** 2)
        observed = shift(observed, -1)
        #observed[obsLength-1] = np.array(Xtt[i]) #using truth
        observed[obsLength-1] = np.array([pred[1], pred[0]])  #using predicted
        #print(observed)
    #print(prediction)
    plt.plot(Xtt[0:obsLength,0], Xtt[0:obsLength,1], marker='x')
    plt.plot(Xtt[obsLength:seqLength,0], Xtt[obsLength:seqLength,1], marker='x')
    plt.plot(prediction[:,0], prediction[:,1], marker='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('dataset_' + str(datatest) + '_' + str(counter) + '.png')
    plt.cla()
    #plt.show()
    #break
    counter += 1
    final_displacement_error += np.linalg.norm(Xtt[i] - pred[0])