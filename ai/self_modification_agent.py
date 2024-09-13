from .self_modifying_ai import SelfModifyingAI

class SelfModificationAgent:
    def __init__(self, name, model_path, initial_code_str=""):
        self.name = name
        self.self_modifying_ai = SelfModifyingAI(model_path, initial_code_str)

    async def process(self, task_details):
        return await self.self_modifying_ai.modify_code(task_details)
