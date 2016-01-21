# Pupose of this code is to iterate over the ImageNet Class Folders and make an ordered consolidated training set of RGB
# images and corresponding signature maps


from __future__ import division
import cPickle as pickle
from datetime import datetime
import os
import sys
import scipy as sp
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
import random
from random import shuffle
from sklearn.externals import joblib

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer


sys.setrecursionlimit(10000)  # for pickle...
np.random.seed(42)
image_size=128
map_size=32

#imagenet_directory='/home/avisek/ImageNet_Stuffs'
imagenet_directory='/home/rs/asantra/VISUAL_SALIENCY/IMAGENET_STUFFS/imagenet_downloader-master'
def load_data():
  imagenet_directory='/home/rs/asantra/VISUAL_SALIENCY/IMAGENET_STUFFS/imagenet_downloader-master'
  folders=[f for f in os.listdir(imagenet_directory) if os.path.isdir(os.path.join(imagenet_directory,f))==True and f[0]=='n']
  RGB_IMAGE_LIST=[]
  MAP_LIST=[]

  for folder in folders[0:140]:  # for small experiment we have iterated over 140 ImageNet classes
    current_folder=os.path.join(imagenet_directory,folder,'Resized_128X128')
  
  
  
  
  

    if os.path.isdir(os.path.join(imagenet_directory,folder,'_Saliency_Map'))==False:
      continue

    saliency_folder_path=os.path.join(imagenet_directory,folder,'_Saliency_Map','Resized_32X32')
    try:
      os.stat(saliency_folder_path)
    except:
      continue;

    images_rgb=[rgb for rgb in os.listdir(current_folder) if os.path.isdir(rgb)==False]
    images_map=[map for map in os.listdir(saliency_folder_path) if os.path.isdir(map)==False]

    for i in range(len(images_rgb)):
      images_rgb[i]=images_rgb[i][:-4]

    for i in range(len(images_map)):
      images_map[i]=images_map[i][:-17]

  
  
    common_images=np.intersect1d(images_map,images_rgb)
    for images in common_images:
      RGB_IMAGE_LIST.append(os.path.join(current_folder,images+'.jpg'))
      MAP_LIST.append(os.path.join(saliency_folder_path,images+'_Saliency_Map.jpg'))
    consolidated_list=zip(RGB_IMAGE_LIST,MAP_LIST)
    shuffle(consolidated_list)
    no_of_examples=len(consolidated_list)
    np.save('no_of_example', no_of_examples)
    np.save('consolidated_list',consolidated_list)
  no_of_examples=np.load('no_of_example.npy')
  
  consolidated_list=np.load('consolidated_list.npy')
  
  
  
  print 'no of examples are:: ', no_of_examples
  X = np.memmap('X4.npy', dtype='float32', mode='w+', shape=(no_of_examples,3,image_size,image_size))
  y = np.memmap('y4.npy', dtype='float32', mode='w+', shape=(no_of_examples,map_size*map_size))
  example=0
  for current_example in range(len(consolidated_list)):
    try:
      rgb=sp.misc.imread(consolidated_list[current_example][0])
      v=rgb/255
      X[example,0,:,:]=v[:,:,0]
      X[example,1,:,:]=v[:,:,1]
      X[example,2,:,:]=v[:,:,2]
      
      gray=sp.misc.imread(consolidated_list[current_example][1])
      gray=gray/255
      y[example,:]=gray.reshape(1,map_size*map_size)
      example=example+1
    except:
      print 'could not process', example
      X[example,0,:,:]=X[example-1,0,:,:]
      X[example,1,:,:]=X[example-1,1,:,:]
      X[example,2,:,:]=X[example-1,2,:,:]
      y[example,:]=y[example-1,:]
      example=example+1
      continue
  return X,y
      
      
      
      
 
