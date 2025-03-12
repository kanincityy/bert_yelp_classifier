import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, test_loader, optimizer, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss()  # Since we have 5 rating classes

    def train(self, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            total_loss = 0

            for batch in loop:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss  # Hugging Face model returns a dict with 'loss'
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Evaluation Accuracy: {accuracy:.4f}")

    def save_model(self, save_path="bert_yelp_model.pth"):
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")