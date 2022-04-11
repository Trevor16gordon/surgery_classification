import torch
import torchvision

import pickle
import cv2
import numpy as np
import pdb
from torchvision.io import read_image
from torchvision import transforms

class SurgeryDataset(torch.utils.data.Dataset):
    """Pytorch dataloader to load single frames at a time.

    Args:
        torch (_type_): _description_
    """

    def __init__(self, list_IDs, labels, base_images_path, data_augmentation=False, n_frames=1, frame_dim=(3,120,200)):
        "Initialization"
        self.labels = labels
        self.list_IDs = list_IDs
        self.base_images_path = base_images_path

        self.frame_dim = frame_dim
        self.n_frames = n_frames

        if data_augmentation:
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.25 if n_frames==1 else 0.0),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p =0.2),
                ttransforms.RandomApply([ransforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)], p=0.2),
                AddGaussianNoise(mean=0.0, std=0.75)
            ]) 
        else:
            self.augment = torch.nn.Identity()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select ID of the final frame in the sample and initialize frames tensor
        start_index, end_index = max(0, index-self.n_frames+1), index+1
        IDs = self.list_IDs[start_index:end_index]
        frames = torch.zeros([self.n_frames, *self.frame_dim])

        label_vid_name = IDs[-1]
        label_vid_name = label_vid_name.split("_vi_")[0] # the videoname of the last frame

        # for each of the previous frames with matching
        offset = abs(min(index-self.n_frames+1, 0)) # handle edge case when index < n_frames
        for i, ID in enumerate(IDs):
            vid_name, frame_id = ID.split("_vi_")
            
            # if the i^th frame is from a previous video, then we leave that frame as zeros
            if vid_name != label_vid_name:
                next

            frame_id = int(frame_id)
            img_path = f"{self.base_images_path}/{vid_name}_frame_{frame_id}.jpg"

            # read in next frame, augment, and add to frames
            X = read_image(img_path)
            X = X.type(torch.FloatTensor)
            X = self.augment(X)
            frames[i + offset] = X
            
        y = self.labels[self.list_IDs[index]]
        return frames.squeeze(), y

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

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
