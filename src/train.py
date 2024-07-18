import torch
from transformers import BertTokenizerFast
from models.bert_model import BertModel
from utils.train_loop import train_loop
from data.panx_loader import PANX_dataloader

import humanize
import datetime as dt

from config.labels import unique_labels
from utils.print_configuration import print_training_config

def train(languages, nrows, model_params, model_path):
    # Load PAN-X NER data
    print("Loading data...")
    dataloader = PANX_dataloader(langs=languages, nrows=nrows)
    df_train, df_val, df_test = dataloader.load_training_data()

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    # Initialize BERT model
    model = BertModel(len(unique_labels))

    # Print training configuration
    print_training_config(languages, nrows, model.bert, tokenizer, model_params)

    # Train model
    print("\nTraining...")
    model, best_model_weights, training_time = train_loop(model, tokenizer, df_train, df_val, model_params)

    natural_training_time = humanize.naturaldelta(dt.timedelta(seconds=training_time))
    print(f"Training took {natural_training_time}")

    # Save weights
    torch.save(best_model_weights, model_path)

languages = ["en", "es"]
model_params = {
        'learning_rate': 3e-3,
        'epochs': 5,
        'batch_size': 2,
    }
nrows = 100
model_path = 'models/pretrained/test_model.pth'
train(languages, nrows, model_params, model_path)