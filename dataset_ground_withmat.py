# The program assumes you have the ground truth images of iSUN, SALICON and MIT according your preferred size (128X128 in our case)
# Also, you have the fixation images stored as mat files

#Output:  a) A consolidated list of ground truth RGB images and corresponding fixation mat files
#          b)Prepares the ground truth by storing RGB and fixation maps at memory mapped .npy files
from __future__ import division
import os,sys
import numpy as np
import scipy as sp
from scipy import misc
import random
from random import shuffle
import h5py

def loading_mat(matfile):
  f=h5py.File(matfile)
  data=f.get('unroll')
  data=np.array(data)
  data=np.transpose(data)
  f.close()
  return data

RGB_IMAGE_LIST=[]
MAP_LIST=[]
image_size=128
map_size=48
ground_truth_directory='/home/rs/asantra/VISUAL_SALIENCY/DATASETS_EYE_FIXATION/'
iSUN_RBG_directory=os.path.join(ground_truth_directory,'iSUN/RGB_128X128')
iSUN_mat_directory=os.path.join(ground_truth_directory,'iSUN/Saliency_Mats48X48')

SALICON_RBG_directory=os.path.join(ground_truth_directory,'SALICON/RGB_128X128')
SALICON_mat_directory=os.path.join(ground_truth_directory,'SALICON/Saliency_Mats48X48')

#MIT_RBG_directory=os.path.join(ground_truth_directory,'MIT/RGB_128X128')
#MIT_map_directory=os.path.join(ground_truth_directory,'MIT/map_32X32')

# Process iSUN IMAGES------------------------------------------
images_rgb_iSUN=[rgb.split('.')[0] for rgb in os.listdir(iSUN_RBG_directory) ]
images_mat_iSUN=[m.split('.')[0] for m in os.listdir(iSUN_mat_directory) ]
common_images_iSUN=np.intersect1d(images_mat_iSUN,images_rgb_iSUN)

for image in common_images_iSUN:
  RGB_IMAGE_LIST.append(os.path.join(iSUN_RBG_directory,image))
  MAP_LIST.append(os.path.join(iSUN_mat_directory,image))
#---------------------------------------------------------------------


# Process SALICON IMAGES------------------------------------------
images_rgb_SALICON=[rgb.split('.')[0] for rgb in os.listdir(SALICON_RBG_directory) ]
images_mat_SALICON=[m.split('.')[0] for m in os.listdir(SALICON_mat_directory) ]
common_images_SALICON=np.intersect1d(images_mat_SALICON,images_rgb_SALICON)

for image in common_images_SALICON:
  RGB_IMAGE_LIST.append(os.path.join(SALICON_RBG_directory,image))
  MAP_LIST.append(os.path.join(SALICON_mat_directory,image))
#---------------------------------------------------------------------


# Process MIT IMAGES------------------------------------------
'''images_rgb_MIT=[rgb for rgb in os.listdir(MIT_RBG_directory) ]
images_map_MIT=[m for m in os.listdir(MIT_map_directory) ]

for i in range(len(images_rgb_MIT)):
  images_rgb_MIT[i]=images_rgb_MIT[i][:-5]

for i in range(len(images_map_MIT)):
  images_map_MIT[i]=images_map_MIT[i][:-11]
common_images_MIT=np.intersect1d(images_map_MIT,images_rgb_MIT)

for image in common_images_MIT:
  RGB_IMAGE_LIST.append(os.path.join(MIT_RBG_directory,image+'.jpeg'))
  MAP_LIST.append(os.path.join(MIT_map_directory,image+'_fixMap.jpg'))
#---------------------------------------------------------------------'''
consolidated_list_ground_truth=zip(RGB_IMAGE_LIST,MAP_LIST)
shuffle(consolidated_list_ground_truth)
no_of_examples_ground_truth=len(consolidated_list_ground_truth)
np.save('no_of_examples_ground_truth_48X48', no_of_examples_ground_truth)
np.save('consolidated_list_ground_truth_48X48',consolidated_list_ground_truth)

print 'no of examples are:: ', no_of_examples_ground_truth
XG = np.memmap('XG_mat_48X48.npy', dtype='float32', mode='w+', shape=(no_of_examples_ground_truth,3,image_size,image_size))
yG = np.memmap('yG_mat_48X48.npy', dtype='float32', mode='w+', shape=(no_of_examples_ground_truth,map_size*map_size))

example=0
for current_example in range(len(consolidated_list_ground_truth)):
  try:
      rgb=sp.misc.imread(consolidated_list_ground_truth[current_example][0]+'.jpg')
      v=rgb/255
      XG[example,0,:,:]=v[:,:,0]
      XG[example,1,:,:]=v[:,:,1]
      XG[example,2,:,:]=v[:,:,2]

      #gray=sp.misc.imread(consolidated_list_ground_truth[current_example][1])
      #gray=gray/255
      yG[example,:]=loading_mat(consolidated_list_ground_truth[current_example][1]+'.mat')
      #yG[example,:]=gray.reshape(1,map_size*map_size)
      example=example+1
  except:
      print 'could not process', example
      XG[example,0,:,:]=XG[example-1,0,:,:]
      XG[example,1,:,:]=XG[example-1,1,:,:]
      XG[example,2,:,:]=XG[example-1,2,:,:]
      yG[example,:]=yG[example-1,:]
      example=example+1
      continue
    
print XG[0,0,:,:]
print yG[0,:]



    
    

