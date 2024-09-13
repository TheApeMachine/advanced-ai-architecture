from transformers import BertTokenizer, BertForSequenceClassification
import torch

class IntentClassifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Adjust num_labels as needed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        
        # Map the predicted class to an intent
        intent_map = {0: 'code_generation', 1: 'self_modify_code', 2: 'complex_task'}  # Adjust as needed
        return intent_map.get(predicted_class, 'unknown')
