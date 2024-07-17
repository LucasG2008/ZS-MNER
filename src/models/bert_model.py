import torch
from transformers import BertForTokenClassification

class BertModel(torch.nn.Module):

    def __init__(self, num_labels):
        """
        Initializes the BertModel class.

        Args:
            num_labels (int): The number of labels for token classification.

        Returns:
            None
        """
        super(BertModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)

    def forward(self, input_id, mask, label):
        """
        Performs the forward pass of the BertModel.

        Args:
            input_id (Tensor): The input tensor of token ids.
            mask (Tensor): The attention mask tensor.
            label (Tensor): The label tensor for token classification.

        Returns:
            output (Tensor): The output tensor from the forward pass.
        """
        output  = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output