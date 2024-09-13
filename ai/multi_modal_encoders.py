# File: ai/multimodal_encoders.py

from transformers import BertModel, BertTokenizer
import torchvision
import torch

class PseudocodeEncoder:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def encode(self, pseudocode):
        inputs = self.tokenizer(pseudocode, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

class DiagramEncoder:
    def __init__(self):
        # Use a CNN for image encoding
        self.model = torchvision.models.resnet18(pretrained=True)

    def encode(self, image):
        # Preprocess image and pass through model
        image = image.unsqueeze(0)  # Add batch dimension
        features = self.model(image)
        return features
