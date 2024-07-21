import torch
from transformers import BertTokenizerFast

from src.models.bert_model import BertModel
from src.utils.evaluate_loop import evaluate_loop

from src.config.labels import unique_labels

def evaluate_model(model_path, testing_data):

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    # Load pre-trained BERT model
    print("\nLoading pretrained model...")
    model = BertModel(len(unique_labels))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate model
    print("\nEvaluating...")
    evaluation_metrics = evaluate_loop(model, tokenizer, testing_data)
    return evaluation_metrics