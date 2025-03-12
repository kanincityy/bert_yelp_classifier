import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from load_data import load_yelp_data
from .dataset import BertYelpDataset
from .yelp_trainer import Trainer
from .model import BertYelpModel


# Load data
train_data, test_data = load_yelp_data()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create dataset instances
train_dataset = BertYelpDataset(train_data['text'], train_data['label'], tokenizer)
test_dataset = BertYelpDataset(test_data['text'], test_data['label'], tokenizer)

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
trainer.train(epochs=3)  # Train for 3 epochs
trainer.evaluate()  # Evaluate on test set
trainer.save_model()  # Save the trained model

print("Training and evaluation complete!")