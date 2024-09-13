# File: ai/agent_base.py

class AgentBase:
    def __init__(self, name):
        self.name = name

    def process(self, task_details):
        raise NotImplementedError("Each agent must implement the process method.")
