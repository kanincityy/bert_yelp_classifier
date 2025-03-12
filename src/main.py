import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from dataset import BertYelpDataset
from trainer import Trainer
from model import BertYelpModel

# Load data
print("Loading data from Hugging Face datasets...")
train_data, test_data = load_dataset("yelp_review_full")["train"], load_dataset("yelp_review_full")["test"]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create dataset instances
train_dataset = BertYelpDataset(train_data, tokenizer)
test_dataset = BertYelpDataset(test_data, tokenizer)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load model
model = BertYelpModel()

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Define Trainer
trainer = Trainer(model, train_loader, test_loader, optimizer)

# Train, evaluate, and save the model
trainer.train(epochs=3, checkpoint_interval=1)  # Train for 3 epochs, save every epoch
trainer.evaluate()  # Evaluate on test set
trainer.save_model()  # Save the final trained model

print("Training and evaluation complete!")