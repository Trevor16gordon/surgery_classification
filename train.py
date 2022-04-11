from data_loader import SurgeryDataset, load_labels, get_surgery_balanced_sampler
from models import SimpleConv, get_transfer_learning_model_for_surgery
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import argparse
import time
import warnings
import shelve
import os
from collections import defaultdict
from datetime import datetime
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
    label_dict, partition = load_labels()
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # data loader parameters
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
            w=1.0,
            batch_size=args.batch_size)
    }
    max_epochs = args.epochs

    # train - validation data Generators
    training_set = SurgeryDataset(partition["train"], label_dict, image_filepath_top, data_augmentation=args.data_aug)
    training_generator = torch.utils.data.DataLoader(training_set, **params_train)

    validation_set = SurgeryDataset(partition["validation"], label_dict, image_filepath_top)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params_val)
    
    n_classes = 14
    model = get_transfer_learning_model_for_surgery(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    
    # Save training metrics for later visualization and degguing. 
    train_loss, train_cm_hist = {}, {}
    now = datetime.now()
    metrics_save_path = os.path.join(
        'training_history', 
        args.model, 
        '_'.join([
            "resample", str(args.class_resampling_weight), # resampling weighting for training data
            'data_aug_' + ('y' if args.data_aug else 'n'), # was data augmentation used
            now.strftime("%d-%m_%H-%M")                    # date-time of start of training
        ])
    ) 


    # ensure that save file is present.
    path = os.path.join('training_history', args.model)
    if not os.path.exists(path):
        os.mkdir(path)

    N = 50 # print evey N mini-batches
    for epoch in range(max_epochs):
        model.train()
        epoch_start, i, running_loss = time.time(), 0, 0.0
        confusion_matrix = torch.zeros([n_classes, n_classes])
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Train on curent mini-batch and store metrics
            outputs = model(local_batch)
            loss = criterion(outputs, local_labels)
            running_loss += loss.item()
            confusion_matrix = update_confusion_matrix(confusion_matrix, outputs, local_labels)
            
            # zero out grads and backprop the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print every N mini-batches
            if i % N == 0:    
                macro_F1, F1, precision, recall = compute_precision_recall_F1(confusion_matrix)
                print(f'Epoch: {epoch + 1}, Mini-Batch:{i :5d}, Mean loss: {running_loss / N:.3f}, Mean Accuracy: {torch.trace(confusion_matrix) / torch.sum(confusion_matrix):.3f}, Macro-F1: {macro_F1:.2f}')
                
                train_loss[i] = running_loss/N
                train_cm_hist[i] = confusion_matrix
                running_loss = 0.0
                
            i += 1

        # Validation
        val_batches, val_loss = 0, 0.0
        confusion_matrix = torch.zeros([n_classes, n_classes])
        model.eval() # evaluate model 
        with torch.no_grad():
            for local_batch, local_labels in validation_generator:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                outputs = model(local_batch)
                val_loss += criterion(outputs, local_labels)
                val_batches += 1
                
                confusion_matrix = update_confusion_matrix(confusion_matrix, outputs, local_labels)

        epoch_time = int(time.time() - epoch_start)
        secs, mins, hrs = epoch_time % 60, (epoch_time // 60) % 60, epoch_time // 3600
        
        macro_F1, F1, precision, recall = compute_precision_recall_F1(confusion_matrix)
        print("Validation Loss: {:2f}\t Validation Accuracy: {:2f}\t Validation Macro-F1: {:.2f}\tRun Time: {:02}:{:02}:{:02}".format(
            val_loss / val_batches, torch.trace(confusion_matrix) / torch.sum(confusion_matrix), macro_F1, hrs, mins, secs
        ))


        with shelve.open(metrics_save_path, writeback=True) as metrics:
            metrics[str(epoch)] = {
                "train_loss": train_loss,
                "train_cm": train_cm_hist,
                "val_loss": val_loss / val_batches,
                "val_cm": confusion_matrix
            }

    if args.save_path != '':
        save(args.save_path, model, optimizer)


def update_confusion_matrix(confusion_matrix, model_output, true_labels):
    """Update the confusion matrix based on the true and predicted labels."""

    preds = torch.argmax(model_output, dim=1)
    for t, p in zip(true_labels.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix

def compute_precision_recall_F1(confusion_matrix):
    """Compute the recall, precision, and F1 score for the current model"""

    recall = torch.diag(confusion_matrix) / (torch.sum(confusion_matrix, axis = 1) + 1e-6)
    precision = torch.diag(confusion_matrix) / (torch.sum(confusion_matrix, axis = 0) + 1e-6)
    F1_score = 2*(precision * recall)/(precision + recall + 1e-6)
    macro_F1 = torch.sum(F1_score) / torch.count_nonzero(F1_score)

    return macro_F1, F1_score, precision, recall


def save(path, network, optim):
    torch.save(
        {
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        }, 
        path
    )

if __name__ == "__main__":
    train()
