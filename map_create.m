% This function returns the Graph Based Visual Saliency Maps on ImageNet images
% input : gbvs_path=path where GBVS library is installed
%       : imagenet_path= path where all ImageNet folders are kept
% Output: saliency maps on ImageNet

function map_create(gbvs_path,imagenet_path)

gbvs_directory=gbvs_path;
imagenet_directory=imagenet_path;
folders=dir(imagenet_directory);

folders=folders(~ismember({folders.name},{'.','..'}));
folders=folders([folders(:).isdir]);
no_of_folders=size(folders,1);

for f=252: no_of_folders
    folder_name=folders(f).name;
    fprintf('Current Folder:: %s, :: Fraction of folders done: %f',folder_name,f/no_of_folders)
    if folder_name(1)~='n'
        continue;
    else
        sub_directory=strcat(imagenet_directory,'/',folder_name);
        cd (sub_directory);
        saliency_folder=strcat(sub_directory,'/_Saliency_Map');
        mkdir (saliency_folder);
        picture_structure=dir('*.jpg');
        
        for pic=1:size(picture_structure,1)
            if picture_structure(pic).bytes/1000<5
                continue;
            else
                current_picture_name=picture_structure(pic).name;
                try 
                	current_picture=imread(current_picture_name);
		catch
			continue;
		end
		current_picture=imresize(current_picture,[256 256]);
                cd (gbvs_directory)
		
                map=gbvs(current_picture);
                map_resized=map.master_map;
		map_resized=imresize(map_resized,[64 64]);
                cd (sub_directory);
                %saliency_folder=strcat(sub_directory,'/_Saliency_Map');
		%mkdir (saliency_folder);
		cd (saliency_folder);
                %saliency_map_name=strcat(sub_directory,'/',saliency_folder,'/',current_picture_name(1:end-4),'_Saliency_Map');
                %mkdir (saliency_folder);
		map_saved=strcat(current_picture_name(1:end-4),'_Saliency_Map.jpg');
		%to_be_saved=strcat(saliency_map_name,'.jpg');
                imwrite(map_resized,map_saved);
		cd (sub_directory);
                
            end
            
        
        end
    end
   
end


% img=imread('/home/avisek/Downloads/gbvs/1.jpg');
% map=gbvs(img);
