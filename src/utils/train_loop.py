import torch
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from data.data_sequence import DataSequence

import copy
from tqdm import tqdm

def train_loop(model, tokenizer, df_train, df_val, model_parameters):
    """
    Trains a model using the given dataset and model parameters.
    
    Args:
        model (torch.nn.Module): The model to be trained.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to preprocess the data.
        df_train (pandas.DataFrame): The training dataset.
        df_val (pandas.DataFrame): The validation dataset.
        model_parameters (dict): The parameters for the model training.
            - learning_rate (float): The learning rate for the optimizer.
            - epochs (int): The number of epochs to train the model.
            - batch_size (int): The batch size for training.
    
    Returns:
        tuple: A tuple containing the trained model and the best model weights.
            - model (torch.nn.Module): The trained model.
            - best_model_weights (dict): The weights of the best model during training.
    """

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
        total_acc_train = 0
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

            for i in range(logits.shape[0]):

                logits_clean = logits[i][train_label[i] != -100]
                label_clean = train_label[i][train_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_train += acc
                total_loss_train += loss.item()

            train_acc_history.append((total_acc_train / ((idx+1) * BATCH_SIZE) ))
            train_loss_history.append((total_loss_train / ((idx+1) * BATCH_SIZE) ))

            loss.backward()
            optimizer.step()

            pbar.set_description(f"[Epoch: {epoch_num + 1}] [Acc: {(total_acc_train / ((idx + 1) * BATCH_SIZE)):.3f}]")

            pbar.update(BATCH_SIZE)

        model.eval()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            pbar = tqdm(total=len(df_val), desc=f"[Validation Acc: {0}]")
            for idx, batch_data in enumerate(val_dataloader):
                val_data, val_label = batch_data
    
                val_label = val_label.to(device)
                mask = val_data['attention_mask'].squeeze(1).to(device)
                input_id = val_data['input_ids'].squeeze(1).to(device)
    
                loss, logits = model(input_id, mask, val_label)
    
                for i in range(logits.shape[0]):
                    logits_clean = logits[i][val_label[i] != -100]
                    label_clean = val_label[i][val_label[i] != -100]
    
                    predictions = logits_clean.argmax(dim=1)
                    acc = (predictions == label_clean).float().mean()
                    total_acc_val += acc
                    total_loss_val += loss.item()
    
                val_acc_history.append((total_acc_val / ((idx+1) * BATCH_SIZE) ))
                val_loss_history.append((total_loss_val / ((idx+1) * BATCH_SIZE) ))

                pbar.set_description(f"[Validation Acc: {(total_acc_val / ((idx + 1) * BATCH_SIZE)):.3f}]")

                pbar.update(BATCH_SIZE)

        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)

        print(
            f'Epochs: {epoch_num+1} \
            | Loss: {total_loss_train / len(df_train): .3f} \
            | Accuracy: {total_acc_train / len(df_train): .3f} \
            | Val Loss: {val_loss: .3f} \
            | Val Accuracy: {val_accuracy: .3f}' \
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
    
    train_acc_history = [tensor.item() if isinstance(tensor, torch.Tensor) else tensor for tensor in train_acc_history]
    train_loss_history = [tensor.item() if isinstance(tensor, torch.Tensor) else tensor for tensor in train_loss_history]

    val_acc_history = [tensor.item() if isinstance(tensor, torch.Tensor) else tensor for tensor in val_acc_history]
    val_loss_history = [tensor.item() if isinstance(tensor, torch.Tensor) else tensor for tensor in val_loss_history]

    plot_acc_loss(train_acc_history, val_acc_history, train_loss_history, val_loss_history)

    return model, best_model_weights

#--------------------------------------------------------------------------------------------------------------#

import seaborn as sns
import matplotlib.pyplot as plt

def plot_acc_loss(train_acc_history, val_acc_history, train_loss_history, val_loss_history):
  """
  Plots the accuracy and loss history of the training process.

  Parameters:
    train_acc_history (list): List of training accuracy values
    val_acc_history (list): List of validation accuracy values
    train_loss_history (list): List of training loss values
    val_loss_history (list): List of validation loss values
  """
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