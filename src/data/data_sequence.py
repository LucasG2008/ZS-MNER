import torch

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer):
        """
        Initialize the DataSequence object.

        Parameters:
        df (DataFrame): The input dataframe containing 'tokens_str' and 'ner_tags_str' columns.
        tokenizer (tokenizer): The tokenizer to process the text data.

        Returns:
        None
        """
        
        txt = df['tokens_str'].values.tolist()
        lb = [i.split() for i in df['ner_tags_str'].values.tolist()]

        # Generate label mappings
        unique_labels = set(lbl for seq in lb for lbl in seq)
        labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}

        self.texts = [tokenizer(str(i), padding='max_length', max_length=512, truncation=True, return_tensors="pt") 
                      for i in txt]
        self.labels = [align_label(i, j, tokenizer, labels_to_ids, label_all_tokens=False) for i,j in zip(txt,lb)]

    def __len__(self):
        return len(self.labels)
    
    def get_batch_data(self, idx):
        return self.texts[idx]
    
    def get_batch_labels(self, idx):
        """
        Get the batch labels for a given index.
        
        Parameters:
        idx (int): The index of the batch labels to retrieve.
        
        Returns:
        torch.Tensor: The batch labels corresponding to the index.
        """
        return torch.LongTensor(self.labels[idx])
    
    def __getitem__(self, idx):
        """
        Get the batch data and labels for a given index.
        
        Parameters:
        idx (int): The index of the batch data and labels to retrieve.
        
        Returns:
        tuple: A tuple containing the batch data and batch labels.
        """
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels
    
    def get_label_mappings(self):
        """
        Generates mappings between unique labels and their corresponding ids.
        Returns two dictionaries: labels_to_ids and ids_to_labels.
        """

        unique_labels = set(lbl for seq in self.ner_tokens for lbl in seq)

        labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
        ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

        return labels_to_ids, ids_to_labels

def align_label(texts, labels, tokenizer, labels_to_ids, label_all_tokens):
    """
    Aligns labels to corresponding word indices based on provided texts, labels, tokenizer, label mappings, and tokenization settings.
    
    Parameters:
    texts (list): The input texts to align labels with.
    labels (list): The labels corresponding to the input texts.
    tokenizer (func): The tokenizer function to tokenize the input texts.
    labels_to_ids (dict): A dictionary mapping labels to their unique ids.
    label_all_tokens (bool): A flag indicating whether to label all tokens or not.
    
    Returns:
    list: The aligned label ids corresponding to the input texts.
    """
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
        previous_word_idx = word_idx
    return label_ids