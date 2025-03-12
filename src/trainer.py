import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import os
import time 

class Trainer:
    def __init__(self, model, train_loader, test_loader, optimizer, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def save_checkpoint(self, epoch, save_path="checkpoints"):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion(self.model(self.input_ids, self.attention_mask, labels=self.labels).logits, self.labels) if hasattr(self,'input_ids') else None,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def train(self, epochs=3, checkpoint_interval=1):
        self.model.train()
        for epoch in range(epochs):
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            total_loss = 0
            start_time = time.time()

            for batch in loop:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.labels = labels

            avg_loss = total_loss / len(self.train_loader)
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch)

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

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {epoch}")
        return epoch