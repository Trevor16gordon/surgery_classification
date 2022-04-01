from data_loader import SurgeryDataset, load_labels



def train():

    video_filepath_top = "/Users/trevorgordon/Library/Mobile Documents/com~apple~CloudDocs/Documents/root/Columbia/Spring2022/AdvancedDL/Assignments/assignment2/surgery_classification/data/videos"

    label_dict, partition = load_labels()
    


    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Parameters
    params = {"batch_size": 64,
            "shuffle": True,
            "num_workers": 6}
    max_epochs = 2

    # Datasets
    partition = # IDs
    labels = # Labels

    # Generators
    training_set = SurgeryDataset(partition["train"], label_dict, video_filepath_top)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = SurgeryDataset(partition["validation"], label_dict, video_filepath_top)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations

        # Validation
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations


if __name__ == "__main__":
    train()