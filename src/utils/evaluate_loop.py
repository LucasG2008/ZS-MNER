import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tqdm import tqdm

from src.data.data_sequence import DataSequence

def evaluate_loop(model, tokenizer, df_test):
    test_dataset = DataSequence(df_test, tokenizer)
    test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
    elif use_cuda:
        model.to(device)
    print(f"Running on: {device}")

    all_preds = []
    all_labels = []

    pbar = tqdm(total=len(df_test), desc=f"[Test Accuracy: {0}]")
    
    for idx, batch_data in enumerate(test_dataloader):
        test_data, test_label = batch_data
        test_label = test_label.to(device)
        
        mask = test_data['attention_mask'].squeeze(1).to(device)
        input_id = test_data['input_ids'].squeeze(1).to(device)

        loss, logits = model(input_id, mask, test_label)

        for i in range(logits.shape[0]):
            logits_clean = logits[i][test_label[i] != -100]
            label_clean = test_label[i][test_label[i] != -100]

            predictions = logits_clean.argmax(dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(label_clean.cpu().numpy())

        # Calculate batch accuracy for display in the progress bar
        acc = (predictions == label_clean).float().mean().item()
        pbar.set_description(f"[Test Accuracy: {acc:.3f}]")
        pbar.update(1)

    # Compute metrics
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average='weighted')
    val_recall = recall_score(all_labels, all_preds, average='weighted')
    val_f1 = f1_score(all_labels, all_preds, average='weighted')

    return {
        'accuracy': val_accuracy,
        'precision': val_precision,
        'recall': val_recall,
        'f1_score': val_f1
    }