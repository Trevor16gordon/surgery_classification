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

def save(df, output_prediction_path):
    df = df.drop("our_id", axis=1)
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    kaggle_temp = pd.read_csv("kaggle_template.csv")
    df = pd.merge(kaggle_temp.drop("Predicted", axis=1), df, left_on="Id", right_on="Id", how="left")
    df.to_csv(output_prediction_path, index=False)


def predict():
    
    parser = argparse.ArgumentParser(description='Specify model and training hyperparameters.')

    parser.add_argument('--input_model_path', type=str)
    parser.add_argument('--output_prediction_path', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_workers', type=int, default=32)
    parser.add_argument('--image_file_path', type=str, default='data/images')
    parser.add_argument('--model', type=str, default='resnet18')

    args = parser.parse_args()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    model = get_transfer_learning_model_for_surgery(args.model)
    checkpoint = torch.load(args.input_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    label_lookup = pd.read_csv("label_lookup.csv")
    label_lookup_dict = {k: v for v, k in zip(label_lookup["label"].tolist(), label_lookup["int"].tolist())}

    df = pd.read_csv("prediction_ids.csv")
    df_save = df.copy()
    df_save = df_save.rename({"target_id": "Id"}, axis=1)
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
    dummy_label_dict = {id:999 for id in pred_ids}

    # Creat a dataloader consisting only of these IDs
    predict_set = SurgeryDataset(pred_ids, dummy_label_dict, args.image_file_path)
    predict_generator = torch.utils.data.DataLoader(predict_set, **predict_params)

    i = 0 
    for images, _ in tqdm.tqdm(predict_generator):
        try:
            images.to(device)
            outputs = model(images)
            size_this_batch = outputs.shape[0]
            preds = torch.argmax(outputs, dim=1)
            pred_numpy = preds.cpu().numpy()
            pred_str = [label_lookup_dict[x] for x in pred_numpy.tolist()]
            df_save["Predicted"].iloc[i: i + size_this_batch] = pred_str
            i += size_this_batch            
        except:
            save(df_save)
            traceback.print_exc()

    save(df_save, args.output_prediction_path)



if __name__ == "__main__":
    predict()

