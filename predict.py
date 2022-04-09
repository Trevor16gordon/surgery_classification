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
from collections import Counter, defaultdict
warnings.filterwarnings("ignore")


def create_list_of_images_to_predict(base_image_path):
    """Required to predict these video ranges
    # All of these videos:
    # Range: RALIHR_surgeon01_fps01_0071 -  RALIHR_surgeon01_fps01_0125 (55 videos)
    # Range: RALIHR_surgeon02_fps01_0001 - RALIHR_surgeon02_fps01_0004 (4 videos)
    # Single: RALIHR_surgeon03_fps01_0001 (1 video)
    """

    if os.path.exists('prediction_ids.csv'):
        with open('prediction_ids.csv', 'r') as csvfile:
            reader = csv.reader(f)
            image_names = list(reader)
    
        idx = []
        file_names = []
        frame_ids = []
        for line in image_names:
            idx.append(line[0])
            file_names.append(os.path.join(base_image_path, line[1]))
            file_ids.append(line[2])
    
    return idx, files_names, frame_ids

    video_names = []
    video_names += [f"RALIHR_surgeon01_fps01_{i:04d}" for i in range(71, 126)]
    video_names += [f"RALIHR_surgeon02_fps01_{i:04d}" for i in range(1, 5)]
    video_names.append("RALIHR_surgeon03_fps01_0001")

    image_names = []
    for vid_name in video_names:
        f =  os.path.join(*base_image_path.split(os.sep), vid_name)
        image_names.extend(glob.glob(f + "*"))

    return image_names


def predict():
    
    parser = argparse.ArgumentParser(description='Specify model and training hyperparameters.')

    parser.add_argument('--input_model_path', type=str)
    parser.add_argument('--output_prediction_path', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_workers', type=int, default=32)
    parser.add_argument('--image_file_path', type=str, default='data/images')

    args = parser.parse_args()

    model = get_transfer_learning_model_for_surgery()
#    checkpoint = torch.load(args.input_model_path)
#   model.load_state_dict(checkpoint['model_state_dict'])

    idx, files_names, frame_ids = create_list_of_images_to_predict(args.image_file_path)
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
    for images, _ in predict_generator:
        images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        for j in range(preds.shape[0]):
            vid_id = pred_ids[i].split('\\')[1].split('_')
            vid_id = f'{int(vid_id[1][-1]):03}-{int(vid_id[2][3:]):04}-{int(vid_id[3]):05}'
            pred_dict[vid_id] = pred[j]

            i += 1

    with open('predictions.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for vid_id, pred in pred_dict.items():
            writer.writerow(vid_id + ', ' + pred)

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
