import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

class BertYelpModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=5):
        super(BertYelpModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
