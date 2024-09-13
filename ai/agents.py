# File: ai/agents.py

from code_generator_ai import CodeGeneratorAI
from coder import Coder

class CodeGenerationAgent(AgentBase):
    def __init__(self, name):
        super().__init__(name)
        self.code_generator_ai = CodeGeneratorAI()
        self.coder = Coder()
        self.experience_buffer = ExperienceReplayBuffer()

    def process(self, task_details):
        # Generate code
        code_ai = self.code_generator_ai.generate_code(task_details)
        code_coder = self.coder.generate(task_details)
        # Store experience
        self.experience_buffer.push({'task': task_details, 'code_ai': code_ai, 'code_coder': code_coder})
        return {'code_ai': code_ai, 'code_coder': code_coder}