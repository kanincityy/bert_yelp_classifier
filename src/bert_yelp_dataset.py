import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class BertYelpDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.reviews = data["text"]
        self.labels = data["label"]
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]

        # Tokenize review text
        tokenized_input = self.tokenizer(
            review,
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

        return {
            'input_ids': torch.tensor(tokenized_input['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(tokenized_input['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }
