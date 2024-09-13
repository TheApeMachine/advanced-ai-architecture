from gliner import GLiNER

class Classifier:
    
    def __init__(self):
        self.model = GLiNER.from_pretrained("numind/NuZero_token")
        self.labels = [
            "language",
            "operation", 
            "numeric",
            "type", 
            "date" 
        ]

    def classify(self, text):
        entities = self.model.predict_entities(text, self.labels)
        return entities
