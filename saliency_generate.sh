#!/bin/sh

gbvs_path="/home/rs/asantra/VISUAL_SALIENCY/SALIENCY_GENERATION/gbvs"

imagenet_path="/home/rs/asantra/VISUAL_SALIENCY/IMAGENET_STUFFS/imagenet_downloader-master"

matlab -r "map_create $gbvs_path $imagenet_path; quit()"
