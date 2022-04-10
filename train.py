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
from collections import Counter
warnings.filterwarnings("ignore")


def train():
    
    parser = argparse.ArgumentParser(description='Specify model and training hyperparameters.')

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_workers', type=int, default=32)
    parser.add_argument('--class_resampling_weight', type=float, default=0.5)
    parser.add_argument('--video_file_path', type=str, default='data/videos')
    parser.add_argument('--image_file_path', type=str, default='data/images')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--data_aug', type=bool, default=False)

    args = parser.parse_args()

    image_filepath_top = args.image_file_path
    # "/Users/trevorgordon/Library/Mobile Documents/com~apple~CloudDocs/Documents/root/Columbia/Spring2022/AdvancedDL/Assignments/assignment2/surgery_classification/data/videos"

    label_dict, partition = load_labels()
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Parameters
    params_train = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.n_workers,
        "sampler": get_surgery_balanced_sampler(
            partition["train"], 
            label_dict, 
            w=args.class_resampling_weight, 
            batch_size=args.batch_size)
    }

    params_val = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.n_workers,
        "sampler": get_surgery_balanced_sampler(
            partition["validation"], 
            label_dict, 
            w=args.class_resampling_weight, 
            batch_size=args.batch_size)
    }
    max_epochs = args.epochs

    # train - validation data Generators
    training_set = SurgeryDataset(partition["train"], label_dict, image_filepath_top)
    training_generator = torch.utils.data.DataLoader(training_set, **params_train)

    validation_set = SurgeryDataset(partition["validation"], label_dict, image_filepath_top)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params_val)
    
    n_classes = 14
    model = get_transfer_learning_model_for_surgery(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    N = 100 # print evey N mini-batches
    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        epoch_start, i = time.time(), 0
        running_loss, N_correct  = 0.0, 0.0
        confusion_matrix = torch.zeros([n_classes, n_classes])

        predict_counts = Counter()
        seen_label_c = Counter()
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Train on curent mini-batch 
            outputs = model(local_batch)
            loss = criterion(outputs, local_labels)
            
            # Keep track of training statistics 
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            for t, p in zip(local_labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            predict_counts += Counter(preds.cpu().numpy())
            seen_label_c += Counter(local_labels.cpu().numpy())
            #print("actual labels seen so far:", seen_label_c, "\tpredicted labels so far: ", predict_counts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            if i == 0: # stats at start of epoch
                print(f'Epoch: {epoch + 1}, Mini-Batch:{i :5d}, Mean loss: {running_loss:.3f}, Mean Accuracy: {torch.trace(confusion_matrix) / torch.sum(confusion_matrix):.3f}')

            if i % N == 0 and 0 < i:    # print every N mini-batches
                print(f'Epoch: {epoch + 1}, Mini-Batch:{i :5d}, Mean loss: {running_loss / N:.3f}, Mean Accuracy: {torch.trace(confusion_matrix) / torch.sum(confusion_matrix):.3f}')
                running_loss, N_correct  = 0.0, 0.0

            i += 1

        # Validation
        val_batchs, val_loss = 0, 0.0
        confusion_matrix = torch.zeros([n_classes, n_classes])
        model.eval() # evaluate model 
        with torch.no_grad():
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                outputs = model(local_batch)

                val_loss += criterion(outputs, local_labels)
                # update confusion matrix
                for t, p in zip(local_labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                val_batchs += 1

        model.train()
        epoch_time = int(time.time() - epoch_start)
        secs = epoch_time % 60
        mins = (epoch_time // 60) % 60
        hrs = epoch_time // 3600

        # calculate precision, recall, & F1
        recall = torch.diag(confusion_matrix) / torch.sum(confusion_matrix, axis = 1)
        precision = torch.diag(confusion_matrix) / torch.sum(confusion_matrix, axis = 0)
        F1_score = 2*(precision * recall)/(precision + recall)

        macro_F1 = torch.mean(F1_score)

        print("Validation Loss: {:2f}\t Validation Accuracy: {:2f}\t Validation Macro-F1: {:.2f}\tRun Time: {:02}:{:02}:{:02}".format(
            val_loss / N, torch.trace(confusion_matrix) / torch.sum(confusion_matrix), macro_F1, hrs, mins, secs
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
