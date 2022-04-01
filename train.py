from data_loader import SurgeryDataset, load_labels
from models import SimpleConv
import torch
import torch.optim as optim
import torch.nn as nn

def train():

    video_filepath_top = "/Users/trevorgordon/Library/Mobile Documents/com~apple~CloudDocs/Documents/root/Columbia/Spring2022/AdvancedDL/Assignments/assignment2/surgery_classification/data/videos"

    label_dict, partition = load_labels()
    

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Parameters
    params = {"batch_size": 6,
            "shuffle": True,
            "num_workers": 1}
    max_epochs = 2


    # Generators
    training_set = SurgeryDataset(partition["train"], label_dict, video_filepath_top)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = SurgeryDataset(partition["validation"], label_dict, video_filepath_top)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    

    model = SimpleConv()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0.0

    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        i = 0
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            print("model computations")
            outputs = model(local_batch)
            loss = criterion(outputs, local_labels)
            loss.backward()
            optimizer.step()


            cur_loss = loss.item()
            print(f"loss is {cur_loss}")

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0

            i += 1

        # # Validation
        # with torch.set_grad_enabled(False):
        #     for local_batch, local_labels in validation_generator:
        #         # Transfer to GPU
        #         local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        #         # Model computations
        #         print("vallidation computations")


if __name__ == "__main__":
    train()