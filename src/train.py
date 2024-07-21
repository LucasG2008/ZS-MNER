import torch
import humanize
import datetime as dt

from transformers import BertTokenizerFast
from src.models.bert_model import BertModel
from src.utils.train_loop import train_loop

from src.config.labels import unique_labels
from src.utils.print_configuration import print_training_config

def train_model(training_data, validation_data, model_params, model_path):

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    # Initialize BERT model
    model = BertModel(len(unique_labels))

    # Print training configuration
    print_training_config(training_data, model.bert, tokenizer, model_params)

    # Train model
    print("\nTraining...")
    model, best_model_weights, training_time = train_loop(model, tokenizer, training_data, validation_data, model_params)

    natural_training_time = humanize.naturaldelta(dt.timedelta(seconds=training_time))
    print(f"Training took {natural_training_time}")

    # Save weights
    torch.save(best_model_weights, model_path)