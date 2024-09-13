from transformers import BertTokenizer, BertForSequenceClassification
import torch

class IntentClassifier:
    def __init__(self, model_path='intent_classifier_model'):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(model_path)

    def classify(self, instruction):
        inputs = self.tokenizer(instruction, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        intents = {0: 'code_generation', 1: 'self_modify_code', 2: 'general_task'}
        return intents[predicted_class_id]
