import torch
from nltk.tokenize import word_tokenize

def evaluate_one_text(model, tokenizer, sentence, ids_to_labels):
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    
    device = torch.device("mps" if use_mps else "cpu")

    if use_mps:
        model.to(device)
    elif use_cuda:
        model = model.cuda()

    print(f"Running on: {device}")

    text = tokenizer(sentence, padding='max_length', max_length=512, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence, tokenizer, label_all_tokens=False)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    display_ner_tags(sentence, prediction_label)

def align_word_ids(texts, tokenizer, label_all_tokens):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx
    return label_ids

def display_ner_tags(sentence, ner_tags):
    tokens = word_tokenize(sentence)
    print("{:<15} {:<8}".format("Word", "NER Tag"))
    print("=" * 25)
    for word, tag in zip(tokens, ner_tags):
        print("{:<15} {:<8}".format(word, tag))