"""This file will process a folder of videos and save to a new folder of images for each frame

It will optionally downsample the images
"""
import argparse
import os
import glob
import cv2
import tqdm
import pdb

parser = argparse.ArgumentParser(description='Specify model and training hyperparameters.')

parser.add_argument('--output_image_width', type=int, default=200)
parser.add_argument('--output_image_height', type=int, default=120)
parser.add_argument('--input_video_folder', type=str, default='data/videos')
parser.add_argument('--output_image_folder', type=str, default='data/images')

args = parser.parse_args()

video_filepath_top = args.input_video_folder
outfolder = args.output_image_folder

new_image_dim = (args.output_image_width, args.output_image_height)


for f in tqdm.tqdm(glob.glob(video_filepath_top + "/*")):
    # Read in the MP4 file from the file of surgery videos
    vidcap = cv2.VideoCapture(f)
    success, image = vidcap.read()
    
    # Extract the name of the file excluding the extension (.mp4) and the file path.
    vid_name = os.path.split(f)[1]
    vid_name = os.path.splitext(vid_name)[0]
    count = 0
    while success:
        resized = cv2.resize(image, new_image_dim)
        image_file = os.path.join(*outfolder.split('/'), f'{vid_name}_frame_{count}.jpg')
        cv2.imwrite(image_file, resized)     # save frame as JPEG file      
        success, image = vidcap.read()
        count += 1
