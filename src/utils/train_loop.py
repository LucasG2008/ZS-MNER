import torch
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader

import sys
import copy
import datetime
from time import time
from tqdm import tqdm

from src.data.data_sequence import DataSequence

def train_loop(model, tokenizer, df_train, df_val, model_parameters):

    start_time = time()

    LEARNING_RATE = model_parameters['learning_rate']
    EPOCHS = model_parameters['epochs']
    BATCH_SIZE = model_parameters['batch_size']

    train_dataset = DataSequence(df_train, tokenizer)
    val_dataset = DataSequence(df_val, tokenizer)

    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()
    elif use_cuda:
        model.to(device)

    print(f"Running on: {device}")

    train_acc_history = []
    train_loss_history = []

    val_acc_history = []
    val_loss_history = []

    #Initialize Variables for EarlyStopping
    best_loss = float('inf')
    best_model_weights = None
    patience = 2

    for epoch_num in range(EPOCHS):

        total_correct_train = 0
        total_samples_train = 0
        total_loss_train = 0

        model.train()

        pbar = tqdm(total=len(df_train), desc=f"[Epoch: {epoch_num + 1}] [Acc: {0}]")
        for idx, batch_data in enumerate(train_dataloader):
            train_data, train_label = batch_data

            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            # Extract labels and predictions from logits
            logits_clean = logits[train_label != -100]
            label_clean = train_label[train_label != -100]

            predictions = logits_clean.argmax(dim=1)
            total_correct_train += (predictions == label_clean).sum().item()
            total_samples_train += label_clean.size(0)
            total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

            # Calculate training accuracy and loss
            train_accuracy = total_correct_train / total_samples_train
            train_loss = total_loss_train / total_samples_train

            train_acc_history.append(train_accuracy)
            train_loss_history.append(train_loss)

            # Update progress bar
            pbar.set_description(f"[Epoch: {epoch_num + 1}] [Acc: {train_accuracy:.3f}]")
            pbar.update(BATCH_SIZE)

        pbar.close()
        sys.stdout.flush()

        model.eval()

        total_correct_val = 0
        total_samples_val = 0
        total_loss_val = 0

        with torch.no_grad():
            pbar = tqdm(total=len(df_val), desc=f"[Validation Acc: {0}]")
            for idx, batch_data in enumerate(val_dataloader):
                val_data, val_label = batch_data
    
                val_label = val_label.to(device)
                mask = val_data['attention_mask'].squeeze(1).to(device)
                input_id = val_data['input_ids'].squeeze(1).to(device)
    
                loss, logits = model(input_id, mask, val_label)

                # Extract labels and predictions from logits
                logits_clean = logits[val_label != -100]
                label_clean = val_label[val_label != -100]

                predictions = logits_clean.argmax(dim=1)
                total_correct_val += (predictions == label_clean).sum().item()
                total_samples_val += label_clean.size(0)
                total_loss_val += loss.item()

                # Calculate validation accuracy and loss
                val_accuracy = total_correct_val / total_samples_val
                val_loss = total_loss_val / total_samples_val

                val_acc_history.append(val_accuracy)
                val_loss_history.append(val_loss)

                # Update progress bar
                pbar.set_description(f"[Validation Acc: {val_accuracy:.3f}]")
                pbar.update(BATCH_SIZE)

        pbar.close()
        sys.stdout.flush()

        # Calculate training and validation metrics
        train_accuracy = total_correct_train / total_samples_train
        train_loss = total_loss_train / total_samples_train
        val_accuracy = total_correct_val / total_samples_val
        val_loss = total_loss_val / total_samples_val

        tqdm.write(
            f'Epochs: {epoch_num+1} \
            | Loss: {train_loss: .3f} \
            | Accuracy: {train_accuracy: .3f} \
            | Val Loss: {val_loss: .3f} \
            | Val Accuracy: {val_accuracy: .3f}'
        )

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here      
            patience = 2  # Reset patience counter
        else:
            patience -= 1
            if patience == 0:
                print('Early Stopping Triggered')
                break

    # Compute training time
    end_time = time()
    training_time = end_time - start_time
    
    # Save and plot training and validation history metrics
    train_acc_history = [tensor.item() if isinstance(tensor, torch.Tensor) else tensor for tensor in train_acc_history]
    train_loss_history = [tensor.item() if isinstance(tensor, torch.Tensor) else tensor for tensor in train_loss_history]

    val_acc_history = [tensor.item() if isinstance(tensor, torch.Tensor) else tensor for tensor in val_acc_history]
    val_loss_history = [tensor.item() if isinstance(tensor, torch.Tensor) else tensor for tensor in val_loss_history]

    plot_acc_loss(train_acc_history, val_acc_history, train_loss_history, val_loss_history)

    return model, best_model_weights, training_time

#--------------------------------------------------------------------------------------------------------------#

import seaborn as sns
import matplotlib.pyplot as plt

def plot_acc_loss(train_acc_history, val_acc_history, train_loss_history, val_loss_history):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

    sns.lineplot(x=range(len(train_acc_history)), y=train_acc_history, ax=ax[0], label='Train Accuracy')
    sns.lineplot(x=[len(train_acc_history)//len(val_acc_history)*i for i in range(len(val_acc_history))], 
                y=val_acc_history, 
                ax=ax[0], 
                label='Validation Accuracy')
    
    ax[0].set_title('Accuracy History')

    sns.lineplot(x=range(len(train_loss_history)), y=train_loss_history, ax=ax[1], label='Train Loss')
    sns.lineplot(x=[len(train_loss_history)//len(val_loss_history)*i for i in range(len(val_loss_history))], 
                y=val_loss_history, 
                ax=ax[1], 
                label='Validation Loss')
    
    ax[1].set_title('Loss History')

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'outputs/plots/training_history_{timestamp}.png')