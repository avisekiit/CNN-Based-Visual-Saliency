# CNN-Based-Visual-Saliency
The projects envisions to use CNNs for visual saliency detection task

Short description of the codes:
1. map_create.m :: creates saliency maps on ImageNet using Graph Based Visual Saliency Model whose library is under the gbvs folder
2. saliency_generate.sh :: a wrapper to call map_create.m with proper arguements
3.startup.m :: file required to include the gbvs library path in Matlab's search path

4. imagenet_saliency_prep.py :: iterates over the entire ImageNet folders to generate a consolidates training set with ImageNet RGB
                                 images and corresponding saliency map
5.dataset_ground_withmat.py:: creates the ground truth training set using eye fixation maps of iSUN, SALICON and MIT datasets
6.final_attempt_pretrain.py:: A Lasagne-Nolearn variant of training CNN on ImageNet saliency maps for pre-training phase
7.final_attempt_posttrain.py:: Uses pre-learned weights to train a CNN model on actual human eye fixation ground truths
8.predict2gray.m : converts the final predicted saliency vectors to gray scale images
------------------------------------
## Some data processing parts have been coded in Matlab because still major competitions on saliency detections upload helper 
functions in Matlab
####################
