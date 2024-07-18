import torch
from torch.utils.data import DataLoader
from data.data_sequence import DataSequence

from tqdm import tqdm

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

    total_acc_test = 0.0

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
            acc = (predictions == label_clean).float().mean()
            total_acc_test += acc

        pbar.set_description(f"[Test Accuracy: {(total_acc_test / ((idx + 1))):.3f}]")

        pbar.update(1)

    val_accuracy = total_acc_test / len(df_test)
    print(f'Test Accuracy: {val_accuracy: .3f}')