import torch
import torchvision

import pickle
import cv2
import numpy as np
import pdb
from torchvision.io import read_image

class SurgeryDataset(torch.utils.data.Dataset):
    """Pytorch dataloader to load single frames at a time.

    Args:
        torch (_type_): _description_
    """

    def __init__(self, list_IDs, labels, base_images_path):
        "Initialization"
        self.labels = labels
        self.list_IDs = list_IDs
        self.base_images_path = base_images_path

        self.to_tensor = torchvision.transforms.ToTensor() 
        self.center_crop = torchvision.transforms.CenterCrop((480, 768))
        self.resize = torchvision.transforms.Resize((120, 192))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        vid_name, frame_id = ID.split("_vi_")
        frame_id = int(frame_id)

        img_path = f"{self.base_images_path}/{vid_name}_frame_{frame_id}.jpg"
        X = read_image(img_path)
        X = X.type(torch.FloatTensor)

        # Size is torch.Size([3, 120, 200])

        y = self.labels[ID]

        return X, y


def load_labels(label_dict_path="data/partition.pkl", partion_path="data/label_dict.pkl"):

      with open(partion_path, "rb") as f:
            label_dict = pickle.load(f)

      with open(label_dict_path, "rb") as f:
            partition = pickle.load(f)

      return label_dict, partition

