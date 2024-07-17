import torch

from transformers import BertTokenizerFast
from models.bert_model import BertModel
from utils.evaluate_loop import evaluate_loop
from data.panx_loader import PANX_dataloader

from config.labels import unique_labels

# Load PAN-X NER data
print("Loading data...")
dataloader = PANX_dataloader(langs=["es", "es"], nrows=1000)
df_test = dataloader.load_data()

# Load tokenizer
print("Loading tokenizer...")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

# Load pre-trained BERT model
print("Loading pretrained model...")
model_path = 'src/models/pretrained/adapter_model.pth'

model = BertModel(len(unique_labels))
model.load_state_dict(torch.load(model_path))
model.eval()

# Evaluate model
print("Evaluating...")
evaluate_loop(model, tokenizer, df_test)