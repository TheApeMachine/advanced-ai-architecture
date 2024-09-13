from .task_decomposer import TaskDecomposer
from .agent_base import AgentBase

class PlanningAgent(AgentBase):
    def __init__(self, name):
        super().__init__(name)
        self.task_decomposer = TaskDecomposer()

    def process(self, task_details):
        subtasks = self.task_decomposer.decompose(task_details)
        return subtasks
