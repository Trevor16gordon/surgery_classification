from data_loader import SurgeryDataset, load_labels
from models import SimpleConv
import torch
import torch.optim as optim
import torch.nn as nn

import argparse

def train():
    
    parser = argparse.ArgumentParser(description='Specify model and training hyperparameters.')

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--video_file_path', type=str, default='data/videos')
    parser.add_argument('--save_path', type=str, default='')

    args = parser.parse_args()

    video_filepath_top = args.video_file_path
    # "/Users/trevorgordon/Library/Mobile Documents/com~apple~CloudDocs/Documents/root/Columbia/Spring2022/AdvancedDL/Assignments/assignment2/surgery_classification/data/videos"

    label_dict, partition = load_labels()
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Parameters
    params = {"batch_size": args.batch_size,
            "shuffle": True,
            "num_workers": args.n_workers}
    max_epochs = args.epochs


    # Generators
    training_set = SurgeryDataset(partition["train"], label_dict, video_filepath_top)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = SurgeryDataset(partition["validation"], label_dict, video_filepath_top)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)
    

    model = SimpleConv(input_dim=(3, 120, 192)).to(device)

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

            # Train on curent mini-batch 
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

        # Validation
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations
                print("vallidation computations")

    if args.save_path != '':
        save(args.save_path, model, optim)


def save(path, network, optim):
    torch.save(
        {
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        }, 
        path
    )

def load(path network, optim):
    checkpoint = torch.load(path)
    network.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    train()
