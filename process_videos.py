"""This file will process a folder of videos and save to a new folder of images for each frame

It will optionally downsample the images
"""
import argparse
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
    vid_name = f.split("/")[-1].split(".")[0]

    vidcap = cv2.VideoCapture(f)

    success, image = vidcap.read()
    

    count = 0
    while success:
        resized = cv2.resize(image, new_image_dim)
        cv2.imwrite(outfolder + f"/{vid_name}_frame_{count}.jpg", resized)     # save frame as JPEG file      
        success, image = vidcap.read()
        count += 1
