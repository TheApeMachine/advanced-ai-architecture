# File: ai/context_manager.py

class ContextManager:
    def __init__(self):
        self.context = ""

    def update_context(self, code_snippet):
        self.context += "\n" + code_snippet

    def get_context(self):
        return self.context

    def reset_context(self):
        self.context = ""
