import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.mlab as mlab

class DataLoad():
	def __init__(self, dir, filename):
		"""
		input:
		- dir: directory where the data is stored
		- filename: name of the data file
		"""
		self.data_path = dir+filename
		self.X = []
		self.X_raw = []
		self.labels = []
		self.data = {}#will be divided into train/val/test
		self.N = 0
		self.iter_train = 0
		self.epochs = 0

	def load_data(self, seq_len = 20, overlap_rate = 0.2, augment = False, verbose = False):
		if self.X:
			print ("You already have the data")
			return
		df = pd.read_csv(self.data_path)
		if verbose:
			print ("the shape of the data is ", df.shape)

		df_arr = df.as_matrix(['x','y','z','t'])
		df = None
		start_idx = 0
		N,D = df_arr.shape#N*4

		df_arr = self.preprocess(df_arr)

		for i in range(1,N,1):
			if verbose and i%10000==0:
				print ("load %5d of %5d"%(i,N))
			if int(df_arr[i,3])==1:#encounter a new sequence
				end_idx = i
				seq = df_arr[start_idx:end_idx,:]
				while seq.shape[0]>=seq_len+1:
					self.X_raw.append(seq[:seq_len,:3])
					self.X.append(self.noisy(seq[:seq_len,:3]))
					self.labels.append(seq[1:seq_len+1,:3])
					if augment:
						self.X_raw.append(self.augment(seq[:seq_len,:3]))
						self.X.append(self.augment(self.noisy(seq[:seq_len,:3])))
						self.labels.append(self.augment(seq[1:seq_len+1,:3]))
					seq = seq[int(seq_len*(1.0 - overlap_rate)):]
				start_idx = end_idx
		if verbose:
			print ("load %d sequences of data"%len(self.X))
		self.X = np.stack(self.X, 0)
		self.X_raw = np.stack(self.X_raw, 0)
		self.labels = np.stack(self.labels, 0)

	def split_train_test(self, train_ratio):
		assert not isinstance(self.X, list), 'first load the data'
		N, seq_len, coords = self.X.shape
		assert seq_len > 1, 'sequence length should be greater than 1'
		assert train_ratio < 1.0, 'invalid train/test ratio'
		idx_cut = int(train_ratio*N)
		self.data['X_train'] = self.X[:idx_cut]
		self.data['y_train'] = self.labels[:idx_cut]
		self.data['X_test'] = self.X[idx_cut:]
		self.data['y_test'] = self.labels[idx_cut:]
		self.data['X_train_raw'] = self.X_raw[:idx_cut]
		self.data['X_test_raw'] = self.X_raw[idx_cut:]
		print ("%d train samples and %d test samples"%(idx_cut, N - idx_cut))

	def preprocess(self, data):
		self.maxZ = np.max(data[:,2])
		print ("max z is %d"%self.maxZ)
		data[:,2] = data[:,2]/self.maxZ
		data[:,0] = data[:,0]/1525
		data[:,1] = data[:,1]/2740
		return data

	def noisy(self, data):
		mean = [0,0,0]
		cov = [[.01/1525,0,0],[0,.01/2740,0],[0,0,.01/self.maxZ]]
		draw = np.random.multivariate_normal(mean, cov, data.shape[0])
		'''tmp = data + draw
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.plot(data[:,0], data[:,1], data[:,2],'r')
		ax.plot(tmp[:,0], tmp[:,1], tmp[:,2],'b')
		ax.set_xlabel('x coordinate')
		ax.set_ylabel('y coordinate')
		ax.set_zlabel('z coordinate')
		plt.show()'''
		return data + draw

	def augment(self, data):
		temp = np.zeros(data.shape)
		temp[:] = data[:]
		temp[:,1] = 1 - data[:,1]
		return temp