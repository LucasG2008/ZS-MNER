import torch
from transformers import BertTokenizerFast

from src.models.bert_model import BertModel
from src.data.panx_loader import PANX_dataloader
from src.utils.evaluate_loop import evaluate_loop
from src.utils.score_model import calculate_model_score

from src.config.labels import unique_labels
from src.config.languages import language_dist, lang_eval_dist

from tqdm import tqdm

def model_performance(model_path, train_data):

    # -------------- Load Evaluation data --------------
    print("\nLoading evaluation data...")
    complete_langs = list(language_dist.keys())
    dataloader = PANX_dataloader(langs=complete_langs, nrows=10000000)
    df_eval = dataloader.load_data(lang_dist=lang_eval_dist)

    # -------------- Load tokenizer --------------
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    # Load pre-trained BERT model
    print("\nLoading pretrained model...")
    model = BertModel(len(unique_labels))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # -------------- Evaluate model --------------
    print("\nEvaluating on all languages...")
    model_performance = {}
    for lang in tqdm(complete_langs):
        # Evaluate model on languages
        evaluation_metrics = evaluate_loop(model, tokenizer, df_eval[df_eval["lang"] == lang])
        # Extract metrics
        lang_accuracy = evaluation_metrics["accuracy"]
        lang_f1 = evaluation_metrics["f1_score"]
        composite_accuracy = (lang_accuracy + lang_f1) / 2
        # Add metrics to dictionary
        model_performance[lang] = composite_accuracy

    # -------------- Extract data usage metrics --------------
    data_usage = train_data['lang'].value_counts().to_dict()

    # -------------- Calculate composite score --------------
    composite_score, acc_score, data_score = calculate_model_score(model_performance, data_usage)

    return composite_score, acc_score, data_score