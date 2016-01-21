# This program computes the final saliency model for mimmicking human visual fixation
# Input : a) Weakly pre-trained CNN model trained on ImageNet
        # b) Actual ground truth human visual fixations
# Output: a) A trained model for saliency prediction in images


import cPickle as pickle
from datetime import datetime
import os
import sys
import scipy
from scipy import misc

from matplotlib import pyplot
import numpy as np
from lasagne import layers
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from sklearn.externals import joblib
from sklearn.base import clone
import scipy.io as sio


'''def load():
    f = file('data_Salicon_T.cPickle', 'rb')
    loaded_obj = pickle.load(f)
    f.close()
    X, y = loaded_obj
    return X, y'''
image_size=128
map_size=32

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]
        tmp =  yb[indices].reshape(bs/2,1,map_size,map_size)
        mirror = tmp[ :,:,:, ::-1]
        yb[indices] =  mirror.reshape(bs/2,map_size*map_size)
        return Xb, yb
'''pretrain_net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('maxout6',layers.FeaturePoolLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, image_size, image_size),
    conv1_num_filters=map_size, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
    conv3_num_filters=64, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
    hidden4_num_units=map_size*map_size*2,
    maxout6_pool_size=2,output_num_units=map_size*map_size,output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.05)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,

    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.05, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    batch_iterator_train=BatchIterator(batch_size=128),
    max_epochs=2,
    verbose=1,
    )

no_of_examples=np.load('no_of_example.npy')
Xtr = np.memmap('X4.npy', dtype='float32', mode='r', shape=(no_of_examples,3,image_size,image_size))
ytr = np.memmap('y4.npy', dtype='float32', mode='r', shape=(no_of_examples,map_size*map_size))
#print 'During training shape of X and y',Xtr.shape, ytr.shape
#pretrain_net.fit(X, y)
#joblib.dump(pretrain_net, 'pretrain_net.pkl')'''
###############################################################################################3
final_net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('maxout6',layers.FeaturePoolLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, image_size, image_size),
    conv1_num_filters=map_size, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
    conv3_num_filters=64, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
    hidden4_num_units=map_size*map_size*2,
    maxout6_pool_size=2,output_num_units=map_size*map_size,output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.05)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,

    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.05, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    batch_iterator_train=BatchIterator(batch_size=128),
    max_epochs=1200,
    verbose=1,
    )
pretrain_net=joblib.load('pretrain_net.pkl')
final_net.load_weights_from(pretrain_net)
no_of_examples_ground_truth=np.load('no_of_examples_ground_truth.npy')

X2=np.memmap('XG_mat.npy', dtype='float32', mode='r', shape=(no_of_examples_ground_truth,3,image_size,image_size))
y2=np.memmap('yG_mat.npy', dtype='float32', mode='r', shape=(no_of_examples_ground_truth,map_size*map_size))
final_net.fit(X2,y2)
joblib.dump(final_net, 'final_net.pkl')

###############################################################################
y=final_net.predict(X2)
sio.savemat('prediction_vector.mat', {'y':y})






