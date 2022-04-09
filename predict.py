from data_loader import SurgeryDataset, load_labels, get_surgery_balanced_sampler
from models import SimpleConv, get_transfer_learning_model_for_surgery
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import argparse
import time
import warnings
import pdb
from collections import Counter
warnings.filterwarnings("ignore")


def predict():
    
    parser = argparse.ArgumentParser(description='Specify model and training hyperparameters.')

    parser.add_argument('--input_model_path', type=str)
    parser.add_argument('--output_prediction_path', type=str)

    args = parser.parse_args()

    model = get_transfer_learning_model_for_surgery()
    checkpoint = torch.load(args.input_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    
    # Load the ID's that we need to predict
    # All of these videos:
    # Range: RALIHR_surgeon01_fps01_0071 -  RALIHR_surgeon01_fps01_0125 (55 videos)
    # Range: RALIHR_surgeon02_fps01_0001 - RALIHR_surgeon02_fps01_0004 (4 videos)
    # Single: RALIHR_surgeon03_fps01_0001 (1 video)


    # Creat a dataloader consisting only of these IDs


    # Predict forward which pull the correct frames
    


    # Change the predict integers to predicted strings
    # You are required to submit a csv file containing two columns: “Id” and “Predicted”
    #   Id: This field represents the frame identifier which is created using the following logic
    #           <surgeon_id [3 chars]>-<sample_id[4 chars]>-<frame_number[5 chars]>
    #           For example: 001-0071-00001
    #               Surgeon_id can be either 001, 002, 003 in our case
    #               Sample_id comes from the last part of file name
    #               Frame_number comes from time * frame_rate thus, it is the number of seconds elapsed from the start of the video as the frame rate of all videos is 1. 
    # Note: The frame numbering starts from 1 instead of 0.
    # All 3 identifier constituents are prepadded with zeros.


    



if __name__ == "__main__":
    predict()
