import json
import torch

from transformers import BertTokenizerFast
from models.bert_model import BertModel
from utils.evaluate_loop import evaluate_loop
from data.panx_loader import PANX_dataloader

from config.labels import unique_labels

def evaluate(model_path, languages, nrows):
    """
    Evaluates a pre-trained BERT model on a given dataset.

    Args:
        model_path (str): The path to the pre-trained BERT model.
        languages (List[str]): The list of languages to load the PAN-X NER data for.
        nrows (int): The number of rows to load from the PAN-X NER data.

    Returns:
        float: The test accuracy of the pre-trained BERT model on the given dataset.

    Raises:
        None

    Description:
        This function loads the PAN-X NER data for the specified languages and number of rows.
        It then loads the BERT tokenizer and pre-trained BERT model.
        The pre-trained BERT model is loaded with the number of unique labels.
        The model is set to evaluation mode.
        The function then evaluates the pre-trained BERT model on the loaded dataset.
        The test accuracy of the model is returned.
    """

    # Load PAN-X NER data
    print("Loading data...")
    dataloader = PANX_dataloader(langs=languages, nrows=nrows)
    df_test = dataloader.load_data()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    # Load pre-trained BERT model
    print("Loading pretrained model...")
    model = BertModel(len(unique_labels))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate model
    print("Evaluating...")
    test_accuracy = evaluate_loop(model, tokenizer, df_test)
    return test_accuracy

model_path = 'src/models/pretrained/var_model.pth'
accuracy_dist_path = 'src/models/pretrained/var_model_accuracy.json'
languages = ["af", "ar", "bg", "bn", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fr", "he", "hi", "hu", "id", "it", "ja", "jv", \
                  "ka", "kk", "ko", "ml", "mr", "ms", "my", "nl", "pt", "ru", "sw", "ta", "te", "th", "tl", "tr", "ur", "vi", "yo", "zh"]
var_rom_accuracy = {}

for idx, language in enumerate(languages):
    print(f"\nEvaluating {language} model {idx+1}/{len(languages)}...")
    test_accuracy = evaluate(model_path, [language], 5)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    var_rom_accuracy[language] = float(test_accuracy)

with open(accuracy_dist_path, 'w') as f:
    json.dump(var_rom_accuracy, f)