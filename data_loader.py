import torch
import torchvision

import pickle
import cv2
import numpy as np
import pdb

class SurgeryDataset(torch.utils.data.Dataset):
    """Pytorch dataloader to load single frames at a time.

    Args:
        torch (_type_): _description_
    """

    def __init__(self, list_IDs, labels, base_video_path):
        "Initialization"
        self.labels = labels
        self.list_IDs = list_IDs
        self.base_video_path = base_video_path

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

        cap = cv2.VideoCapture(self.base_video_path + "/" + vid_name + ".mp4" )
        cap.set(1,frame_id)
        ret, frame = cap.read() 

        X = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        X = X.astype(np.float32)
        X = self.to_tensor(X)
        X = self.center_crop(X)
        X = self.resize(X)
            
        y = self.labels[ID]

        return X, y


def load_labels(label_dict_path="data/partition.pkl", partion_path="data/label_dict.pkl"):

      with open(partion_path, "rb") as f:
            label_dict = pickle.load(f)

      with open(label_dict_path, "rb") as f:
            partition = pickle.load(f)

      return label_dict, partition

