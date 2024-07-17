import torch
from transformers import BertTokenizerFast
from models.bert_model import BertModel
from utils.evaluate_one_text import evaluate_one_text

from config.labels import unique_labels, ids_to_labels

# Load tokenizer
print("Loading tokenizer...")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

# Load pre-trained BERT model
model_path = 'src/models/pretrained/var_model.pth'

model = BertModel(len(unique_labels))
model.load_state_dict(torch.load(model_path))
model.eval()

# Sentence to predict NER tags for
sentence = "In 2023, Elon Musk met with executives from Google at their headquarters in Mountain View, California, to discuss advancements in artificial intelligence."

evaluate_one_text(model, tokenizer, sentence, ids_to_labels)