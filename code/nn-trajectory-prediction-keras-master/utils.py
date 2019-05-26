import os
import pickle
import numpy as np
import tensorflow as tf
from keras import backend as K
import random

def mean_squared_error(y_true, y_pred):
    print('mse buatan', y_true.shape, y_pred.shape)
    return K.mean(K.square(y_pred - y_true), axis=-1)


def negative_log_likelihood_w(point, distribution):
    print('using negative_log_likelihood by wolfram')
    # http://mathworld.wolfram.com/BivariateNormalDistribution.html
    '''mux = distribution[:, :, 0]
    muy = distribution[:, :, 1]
    sx = distribution[:, :, 2]
    sy = distribution[:, :, 3]
    rho = distribution[:, :, 4]
    x = point[:, :, 1]
    y = point[:, :, 0]
    '''
    mux = distribution[:, 0]
    muy = distribution[:, 1]
    sx = distribution[:, 2]
    sy = distribution[:, 3]
    rho = distribution[:, 4]
    x = point[:, 1]
    y = point[:, 0]

    z1 = tf.div(K.square(tf.subtract(x, mux)), K.square(sx))
    z2 = tf.div(tf.multiply(2., tf.multiply(rho, tf.multiply(tf.subtract(x, mux), tf.subtract(y, muy)))), tf.multiply(sx, sy))
    z3 = tf.div(K.square(tf.subtract(y, muy)), K.square(sy))
    z = tf.add(tf.add(z1, -z2), z3)

    square_root = K.sqrt(tf.subtract(1., K.square(rho)))
    p1 = tf.div(1., tf.multiply(2., tf.multiply(np.pi, tf.multiply(sx, tf.multiply(sy, square_root)))))
    p2 = K.exp(tf.div(-z, tf.multiply(2., tf.subtract(1., K.square(rho)))))

    p = tf.multiply(p1, p2)

    #log = -tf.log(p)
    epsilon = 1e-20
    log = -tf.log(tf.maximum(epsilon, p))

    result = K.sum(log)
    return result


def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
    '''
    Function that implements the PDF of a 2D normal distribution
    params:
    x : input x points
    y : input y points
    mux : mean of the distribution in x
    muy : mean of the distribution in y
    sx : std dev of the distribution in x
    sy : std dev of the distribution in y
    rho : Correlation factor of the distribution
    '''
    # eq 3 in the paper
    # and eq 24 & 25 in Graves (2013)
    # Calculate (x - mux) and (y-muy)
    normx = tf.subtract(x, mux)
    normy = tf.subtract(y, muy)
    # Calculate sx*sy
    sxsy = tf.multiply(sx, sy)
    # Calculate the exponential factor
    z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) - 2*tf.div(tf.multiply(rho, tf.multiply(normx, normy)), sxsy)
    negRho = 1 - tf.square(rho)
    # Numerator
    result = tf.exp(tf.div(-z, 2*negRho))
    # Normalization constant
    denom = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negRho))
    # Final PDF calculation
    result = tf.div(result, denom)
    return result
    return muy


def negative_log_likelihood(y_true, y_pred):
    print('negative_log_likelihood', y_true.shape, y_pred.shape)
    #y_pred = K.print_tensor(y_pred, 'y_pred ')
    #y_true = K.print_tensor(y_true, 'y_true ')

    '''z_mux = y_pred[:, :, 0]
    z_muy = y_pred[:, :, 1]
    z_sx = y_pred[:, :, 2]
    z_sy = y_pred[:, :, 3]
    z_corr = y_pred[:, :, 4]
    x_data = y_true[:, :, 1]
    y_data = y_true[:, :, 0]
    '''
    z_mux = y_pred[:, 0]
    z_muy = y_pred[:, 1]
    z_sx = K.exp(y_pred[:, 2])
    z_sy = K.exp(y_pred[:, 3])
    z_corr = K.tanh(y_pred[:, 4])

    x_data = y_true[:, 0]
    y_data = y_true[:, 1]

    z_mux = K.print_tensor(z_mux, 'z_mux ')
    z_muy = K.print_tensor(z_muy, 'z_muy ')
    z_sx = K.print_tensor(z_sx, 'z_sx ')
    z_sy = K.print_tensor(z_sy, 'z_sy ')
    z_corr = K.print_tensor(z_corr, 'z_corr ')

    x_data = K.print_tensor(x_data, 'x_data ')
    y_data = K.print_tensor(y_data, 'y_data ')

    step = tf.constant(1e-3, dtype=tf.float32, shape=(1, 1))

    # Calculate the PDF of the data w.r.t to the distribution
    result0_1 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
    #result0_1 = K.print_tensor(result0_1, 'result0_1 ')
    result0_2 = tf_2d_normal(tf.add(x_data, step), y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
    result0_3 = tf_2d_normal(x_data, tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)
    result0_4 = tf_2d_normal(tf.add(x_data, step), tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)

    result0 = tf.div(tf.add(tf.add(tf.add(result0_1, result0_2), result0_3), result0_4),
                     tf.constant(4.0, dtype=tf.float32, shape=(1, 1)))
    result0 = tf.multiply(tf.multiply(result0, step), step)

    # For numerical stability purposes
    epsilon = 1e-20

    # TODO: (resolve) I don't think we need this as we don't have the inner
    # summation
    # result1 = tf.reduce_sum(result0, 1, keep_dims=True)
    # Apply the log operation
    result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability
    #result1 = -tf.log(tf.maximum(result0, epsilon))

    # TODO: For now, implementing loss func over all time-steps
    # Sum up all log probabilities for each data point
    result = tf.reduce_sum(result1)
    result = K.print_tensor(result, 'result')
    return result

def negative_log_likelihood_sequences(y_true, y_pred):
    print('negative_log_likelihood_sequences', y_true.shape, y_pred.shape)
    #y_pred = K.print_tensor(y_pred, 'y_pred ')
    #y_true = K.print_tensor(y_true, 'y_true ')

    z_mux = y_pred[:, :, 0]
    z_muy = y_pred[:, :, 1]
    z_sx = K.exp(y_pred[:, :, 2])
    z_sy = K.exp(y_pred[:, :, 3])
    z_corr = K.tanh(y_pred[:, :, 4])
    x_data = y_true[:, :, 0]
    y_data = y_true[:, :, 1]
    '''    
    z_mux = y_pred[:, 0]
    z_muy = y_pred[:, 1]
    z_sx = y_pred[:, 2]
    z_sy = y_pred[:, 3]
    z_corr = y_pred[:, 4]

    x_data = y_true[:, 1]
    y_data = y_true[:, 0]'''

    '''z_mux = K.print_tensor(z_mux, 'z_mux ')
    z_muy = K.print_tensor(z_muy, 'z_muy ')
    z_sx = K.print_tensor(z_sx, 'z_sx ')
    z_sy = K.print_tensor(z_sy, 'z_sy ')
    z_corr = K.print_tensor(z_corr, 'z_corr ')

    x_data = K.print_tensor(x_data, 'x_data ')
    y_data = K.print_tensor(y_data, 'y_data ')'''

    step = tf.constant(1e-3, dtype=tf.float32, shape=(1, 1))

    # Calculate the PDF of the data w.r.t to the distribution
    result0_1 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
    #result0_1 = K.print_tensor(result0_1, 'result0_1 ')
    result0_2 = tf_2d_normal(tf.add(x_data, step), y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
    result0_3 = tf_2d_normal(x_data, tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)
    result0_4 = tf_2d_normal(tf.add(x_data, step), tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)

    result0 = tf.div(tf.add(tf.add(tf.add(result0_1, result0_2), result0_3), result0_4),
                     tf.constant(4.0, dtype=tf.float32, shape=(1, 1)))
    result0 = tf.multiply(tf.multiply(result0, step), step)

    # For numerical stability purposes
    epsilon = 1e-20

    # TODO: (resolve) I don't think we need this as we don't have the inner
    # summation
    # result1 = tf.reduce_sum(result0, 1, keep_dims=True)
    # Apply the log operation
    result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability
    #result1 = -tf.log(tf.maximum(result0, epsilon))

    # TODO: For now, implementing loss func over all time-steps
    # Sum up all log probabilities for each data point
    return tf.reduce_sum(result1)
    #return tf.reduce_sum(result1)


def sample_gaussian_2d(predicted_params):
    '''
    Function to sample a point from a given 2D normal distribution
    params:
    mux : mean of the distribution in x
    muy : mean of the distribution in y
    sx : std dev of the distribution in x
    sy : std dev of the distribution in y
    rho : Correlation factor of the distribution
    '''
    mux = predicted_params[0,0]
    muy = predicted_params[0,1]
    sx = predicted_params[0,2]
    sy = predicted_params[0,3]
    rho = predicted_params[0,4]
    # Extract mean
    mean = [mux, muy]
    # Extract covariance matrix
    cov = [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]]
    # Sample a point from the multivariate normal distribution
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def shift(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = 0
        e[n:] = xs[:-n]
    else:
        e[n:] = 0
        e[:n] = xs[-n:]
    return e

class DataLoader():

    def __init__(self, batch_size=50, seq_length=5, datasets=[0, 1, 2, 3, 4], forcePreProcess=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : RNN sequence length
        '''
        # List of data directories where raw data resides
        self.data_dirs = ['train_data\\eth_eth', 'train_data\\eth_hotel',
                          'train_data\\ucy_zara01', 'train_data\\ucy_zara02',
                          'train_data\\ucy_ucy']
        #self.data_dirs = ['train_data\\eth_eth', 'train_data\\eth_hotel']

        self.used_data_dirs = [self.data_dirs[x] for x in datasets]

        # Data directory where the pre-processed pickle file resides
        self.data_dir = 'train_data'

        # Store the batch size and the sequence length arguments
        self.batch_size = batch_size
        self.seq_length = seq_length
        # Define the path of the file in which the data needs to be stored
        data_file = os.path.join(self.data_dir, "trajectories.cpkl")

        # If the file doesn't exist already or if forcePreProcess is true
        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files
            self.preprocess(self.used_data_dirs, data_file)

        # Load the data from the pickled file
        #self.load_preprocessed(data_file)
        # Reset all the pointers
        #self.reset_batch_pointer()

    def preprocess(self, data_dirs, data_file):
        '''
        The function that pre-processes the pixel_pos.csv files of each dataset
        into data that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        # all_ped_data would be a dictionary with mapping from each ped to their
        # trajectories given by matrix 3 x numPoints with each column
        # in the order x, y, frameId
        # Pedestrians from all datasets are combined
        # Dataset pedestrian indices are stored in dataset_indices # apa artinya ?
        all_ped_data = {}
        dataset_indices = []
        current_ped = 0
        ped_data_indices = {} # dataset indices for each pedestrian
        count_dataset = 0

        # For each dataset
        for directory in data_dirs:
            # Define the path to its respective csv file
            file_path = os.path.join(directory, 'pixel_pos.csv')

            # Load data from the csv file
            # Data is a 4 x numTrajPoints matrix
            # where each column is a (frameId, pedId, y, x) vector
            data = np.genfromtxt(file_path, delimiter=',')

            # Get the number of pedestrians in the current dataset
            numPeds = np.size(np.unique(data[1, :]))

            # For each pedestrian in the dataset
            for ped in range(1, numPeds+1):
                # Extract trajectory of the current ped
                traj = data[:, data[1, :] == ped]
                # Format it as (x, y, frameId)
                traj = traj[[3, 2, 0], :]

                # Store this in the dictionary
                all_ped_data[current_ped + ped] = traj
                ped_data_indices[current_ped + ped] = count_dataset

            # Current dataset done
            dataset_indices.append(current_ped+numPeds)
            current_ped += numPeds

            count_dataset += 1

        # The complete data is a tuple of all pedestrian data, and dataset ped indices
        complete_data = (all_ped_data, dataset_indices, ped_data_indices)
        # Store the complete data into the pickle file
        f = open(data_file, "wb")
        pickle.dump(complete_data, f, protocol=2)
        f.close()

    def get_data(self, test_dataset):
        data_file = os.path.join(self.data_dir, "trajectories.cpkl")
        # Load data from the pickled file
        f = open(data_file, "rb")
        self.raw_data = pickle.load(f)
        f.close()

        # Get the pedestrian data from the pickle file
        all_ped_data = self.raw_data[0]
        # Not using dataset_indices for now
        #dataset_indices = self.raw_data[1]

        ped_data_indices = self.raw_data[2]
        # Construct the data with sequences(or trajectories)
        data_train = []
        data_test = []
        # For each pedestrian in the data
        for ped in all_ped_data:
            # Extract his trajectory
            traj = np.float32(all_ped_data[ped])
            if traj.shape[1] < self.seq_length:
                continue
            for i in range(0, traj.shape[1] - (self.seq_length - 1)):
                # TODO: (Improve) Store only the (x,y) coordinates for now
                if ped_data_indices[ped] != test_dataset:
                    data_train.append(traj[[0, 1], i:i+self.seq_length].T)
                elif ped_data_indices[ped] == test_dataset:
                    data_test.append(traj[[0, 1], i:i+self.seq_length].T)
                else:
                    print('nothing ', ped_data_indices[ped])
        return data_train, data_test

