% function to convert predicted saliency vectors to saliency maps

function predict2gray()

%MIT_RGBTEST_PATH='/home/rs/asantra/VISUAL_SALIENCY/DATASETS_EYE_FIXATION/MIT300/BenchmarkIMAGES/'
%MIT_RGBTEST_PATH='/home/rs/asantra/VISUAL_SALIENCY/DATASETS_EYE_FIXATION/MIT/RGB_128X128/'
MIT_SALIENCY_TEST_PATH='/home/rs/asantra/VISUAL_SALIENCY/DATASETS_EYE_FIXATION/MIT300/SALIENCY_MAPS/'
%MIT_SALIENCY_TEST_PATH='/home/rs/asantra/VISUAL_SALIENCY/DATASETS_EYE_FIXATION/MIT300/SALIENCY_MAPS/48X48';
mkdir (MIT_SALIENCY_TEST_PATH);


load('prediction_vector_posttrain_net_dropout_300_it.mat');
cd (MIT_SALIENCY_TEST_PATH):
size(y,1)
for i=1:size(y,1)
  predict_vector=y(i,:);
%  size=XSIZE(i,:);
  predict_image=mat2gray(reshape(predict_vector,[32,32]));
  predict_image=imresize(predict_image,[512,512]); % you can resize to any size you want for comparison
  %if i>297
   % imwrite(predict_image,strcat('i',int2str(i),'Waldo.jpg'));
  %else
    imwrite(predict_image,strcat('i',int2str(i),'.jpg'));
 % end
end

exit(0)
