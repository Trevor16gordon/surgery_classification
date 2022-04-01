from data_loader import SurgeryDataset, load_labels


def train():

    video_filepath_top = "/Users/trevorgordon/Library/Mobile Documents/com~apple~CloudDocs/Documents/root/Columbia/Spring2022/AdvancedDL/Assignments/assignment2/surgery_classification/data/videos"

    label_dict, partition = load_labels()
    training_set = SurgeryDataset(partition['train'], label_dict, video_filepath_top)


if __name__ == "__main__":
    train()