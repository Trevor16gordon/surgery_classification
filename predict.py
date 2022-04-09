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
import glob
import os
import csv
import traceback
import pandas as pd
from collections import Counter, defaultdict
warnings.filterwarnings("ignore")



def predict():
    
    parser = argparse.ArgumentParser(description='Specify model and training hyperparameters.')

    parser.add_argument('--input_model_path', type=str)
    parser.add_argument('--output_prediction_path', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_workers', type=int, default=32)
    parser.add_argument('--image_file_path', type=str, default='data/images')

    args = parser.parse_args()
    

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    model = get_transfer_learning_model_for_surgery()
    checkpoint = torch.load(args.input_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    label_lookup = pd.read_csv("label_lookup.csv")
    label_lookup_dict = {k: v for v, k in zip(label_lookup["label"].tolist(), label_lookup["int"].tolist())}

    df = pd.read_csv("prediction_ids.csv")
    df_save = df.copy()
    df_save.rename({"target_id": "Id"}, axis=1)
    df_save["Predicted"] = ""
    pred_ids = df["our_id"].tolist()
    target_ids = df["target_id"].tolist()

    target_id_lookup = {k:v for k, v in zip(pred_ids, target_ids)}

    predict_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.n_workers,
    }

    # Using a dummy dict for labels since we don't have that and the dataloader wants it
    dumy_label_dict = defaultdict(lambda: 99999)

    # Creat a dataloader consisting only of these IDs
    predict_set = SurgeryDataset(pred_ids, dumy_label_dict, args.image_file_path)
    predict_generator = torch.utils.data.DataLoader(predict_set, **predict_params)

    
    pred_dict = {}
    i = 0 
    for images, _ in tqdm.tqdm(predict_generator):
        try:
            images.to(device)
            outputs = model(images)
            size_this_batch = outputs.shape[0]
            preds = torch.argmax(outputs, dim=1)
            #pdb.set_trace()
            pred_numpy = preds.cpu().numpy()
            pred_str = [label_lookup_dict[x] for x in pred_numpy.tolist()]
            df_save["Predicted"].iloc[i: i + size_this_batch] = pred_str
            i += size_this_batch
            
        except:
            df_save = df_save.drop("our_id", axis=1)
            df_save.to_csv("predictions.csv")
            traceback.print_exc()

    df_save = df_save.drop("our_id", axis=1)
    if "Unnamed: 0" in df_save.columns:
        df_save = df_save.drop("Unnamed: 0", axis=1)
    df_save.to_csv("predictions.csv", index=False)

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

