from data_loader import SurgeryDataset, load_labels
from models import SimpleConv
import torch
import torch.optim as optim
import torch.nn as nn

import argparse
import time
import warnings
warnings.filterwarnings("ignore")


def train():
    
    parser = argparse.ArgumentParser(description='Specify model and training hyperparameters.')

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--video_file_path', type=str, default='data/videos')
    parser.add_argument('--image_file_path', type=str, default='data/images')
    parser.add_argument('--save_path', type=str, default='')

    args = parser.parse_args()

    image_filepath_top = args.image_file_path
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
    training_set = SurgeryDataset(partition["train"], label_dict, image_filepath_top)


    # training_set.__getitem__(1)

    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = SurgeryDataset(partition["validation"], label_dict, image_filepath_top)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)
    

    # model = SimpleConv(input_dim=(3, 120, 192)).to(device)
    model = SimpleConv(input_dim=(3, 120, 200)).to(device)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    N = 100 # print evey N mini-batches
    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        epoch_start, i = time.time(), 1
        running_loss, N_correct  = 0.0, 0.0
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Train on curent mini-batch 
            outputs = model(local_batch)
            loss = criterion(outputs, local_labels)

            # Keep track of training statistcis 
            running_loss += loss.item()
            N_correct += torch.sum(torch.argmax(outputs, dim=1) == local_labels)

            loss.backward()
            optimizer.step()

            # print statistics
            if i % N == 0:    # print every N mini-batches
                print(f'Epoch: {epoch + 1}, Mini-Batch:{i :5d}, Mean loss: {running_loss / N:.3f}, Mean Accuracy: {N_correct / (args.batch_size*N):.3f}')
                running_loss, N_correct  = 0.0, 0.0

            i += 1

        # Validation
        val_batchs, val_loss, val_correct = 0, 0.0, 0.0
        model.eval() # evaluate model 
        with torch.no_grad():
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                outputs = model(local_batch)

                val_correct += torch.sum(torch.argmax(outputs, dim=1) == local_labels)
                val_loss += criterion(outputs, local_labels)
                val_batchs += 1

        model.train()
        epoch_time = int(time.time() - epoch_start)
        secs = epoch_time % 60
        mins = (epoch_time // 60) % 60
        hrs = epoch_time // 3600

        print("Validation Loss: {:2f}\t Validation Accuracy: {:2f}\t Run Time: {:02}:{:02}:{:02}".format(
            val_loss / N, val_correct / (args.batch_size*val_batchs), hrs, mins, secs
        ))

    if args.save_path != '':
        save(args.save_path, model, optimizer)


def save(path, network, optim):
    torch.save(
        {
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        }, 
        path
    )

def load(path, network, optim):
    checkpoint = torch.load(path)
    network.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    train()
