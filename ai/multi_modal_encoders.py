import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class PseudocodeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def encode(self, pseudocode):
        inputs = self.tokenizer(pseudocode, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert(**inputs)
        return outputs.last_hidden_state

class DiagramEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 768)  # Assuming input image size of 224x224
        )

    def encode(self, diagram):
        # Assuming diagram is a tensor of shape (batch_size, 3, 224, 224)
        return self.cnn(diagram)
