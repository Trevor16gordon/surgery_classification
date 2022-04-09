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


def get_surgery_balanced_sampler(list_IDs, label_dict, w=0.5, batch_size=64):
    """Get a pytorch sampler to correct unbalanced classes

    This function returns a sampler that is in between an
        uncorrected sampler: If you just pulled random samples from the dataset
        corrected sampler: Sample the under-represented classes at a much higher frequency
            to achieve perfectly balanced classes

        w = 1 for no balancing
        w = 0 for a perfectly corrected sampler
        w = 0.5 for the middle of the two

    The return torch.utils.data.sampler.WeightedRandomSampler can be passed to 
    a pytorch torch.utils.data.DataLoader

    Args:
        w (float, optional): Weighting for the unblanced sampler. Defaults to 0.5.

    Returns:
        torch.utils.data.sampler.WeightedRandomSampler: The sampler
    """

    class_occurance = {
        0: 0.001371, 
        1: 0.000423,
        2: 0.013461, 
        3: 0.129151, 
        4: 0.005417, 
        5: 0.007132, 
        6: 0.260689, 
        7: 0.050488, 
        8: 0.067875, 
        9: 0.133452, 
        10: 0.005314, 
        11: 0.239680, 
        12: 0.010197, 
        13: 0.075342,
    }

    sampling = {k: 1/v for k, v in class_occurance.items()}
    total_sum = sum(sampling.values())
    sampling = {k: v/total_sum for k, v in sampling.items()}

    sampling_weights_uncorrected = np.array(list(class_occurance.values()))
    sampling_weights_corrected = np.array(list(sampling.values()))


    sampling_weights = w*sampling_weights_uncorrected + (1-w)*sampling_weights_corrected
    sampling_weights_cls = sampling_weights / np.sum(sampling_weights)


    all_w = [sampling_weights_cls[label_dict[x]] for x in list_IDs]

    all_w_torch = torch.DoubleTensor(all_w)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(all_w_torch, len(list_IDs))
    return sampler
