import time
import torch
from transformers import BertTokenizerFast
from models.bert_model import BertModel
from utils.train_loop import train_loop
from data.panx_loader import PANX_dataloader

from config.labels import unique_labels

# Load PAN-X NER data
print("Loading data...")
languages = ["en", "es"]
dataloader = PANX_dataloader(langs=languages, nrows=100)
df_train, df_val, df_test = dataloader.load_training_data()

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

# Initialize BERT model
model = BertModel(len(unique_labels))

# Specify model parameters
model_params = {
    'learning_rate': 3e-3,
    'epochs': 5,
    'batch_size': 2,
}

start = time.time()

# Train model
print("\nTraining...")
model, best_model_weights = train_loop(model, tokenizer, df_train, df_val, model_params)

end = time.time()
print(f"Training took {end - start} seconds")

# Save weight
#model_path = 'src/models/pretrained/en_es_model.pth'
#torch.save(best_model_weights, model_path)