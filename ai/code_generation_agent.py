from .code_generator_ai import CodeGeneratorAI
from .coder import Coder
from .agent_base import AgentBase
from .experience_replay import ExperienceReplayBuffer

class CodeGenerationAgent(AgentBase):
    def __init__(self, name, model_path):
        super().__init__(name)
        self.code_generator_ai = CodeGeneratorAI(model_path)
        self.coder = Coder()
        self.experience_buffer = ExperienceReplayBuffer()

    def process(self, task_details):
        # Generate code
        code_ai = self.code_generator_ai.generate_code(task_details)
        code_coder = self.coder.generate(task_details)
        # Store experience
        self.experience_buffer.push({'task': task_details, 'code_ai': code_ai, 'code_coder': code_coder})
        return {'code_ai': code_ai, 'code_coder': code_coder}