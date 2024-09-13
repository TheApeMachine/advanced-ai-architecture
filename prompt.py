from typing import List, Dict, Any
import json

class Prompt:
    def __init__(self):
        self.system_context = """
        You are part of an advanced AI system composed of multiple Expert agents. 
        Each Expert, including yourself, has the ability to perform various roles and tasks based on the given prompt. 
        You have access to various tools, including the ability to create and communicate with assistant Experts when necessary.
        When you identify multiple tasks or a complex task that can be broken down, use the task_breakdown tool to create a queue of subtasks.
        Your responses should be in JSON format for easy parsing and integration with other system components.
        """
        self.role_templates = {
            "general": "You are a general-purpose Expert capable of handling a wide range of tasks.",
            "manager": "You are currently acting in a managerial capacity, responsible for task delegation, system oversight, and decision-making.",
            "specialist": "You are a specialist Expert focused on {domain}. Your primary responsibility is {responsibility}."
        }
        self.task_history = []
        self.system_state = {}

    def generate_prompt(self, role, task, context):
        role_template = self.role_templates.get(role, self.role_templates["general"])
        prompt = f"{self.system_context}\n\n{role_template}\n\nTask: {task}\n\nContext: {json.dumps(context)}\n\nResponse:"
        return prompt

    def update_task_history(self, task: str, outcome: str):
        self.task_history.append({"task": task, "outcome": outcome})
        if len(self.task_history) > 20:  # Keep only the last 20 tasks
            self.task_history.pop(0)

    def update_system_state(self, state: Dict[str, Any]):
        self.system_state.update(state)

    def generate_reflection_prompt(self) -> str:
        reflection_prompt = f"""
        {self.system_context}

        Please reflect on the recent task history and current system state to improve overall performance.

        Recent task history: {json.dumps(self.task_history)}
        Current system state: {json.dumps(self.system_state)}

        Provide your reflection in the following JSON format:
        {{
            "insights": "Key insights from recent tasks and system state",
            "improvement_suggestions": "Suggestions for system improvement",
            "proposed_changes": "Specific changes to prompts, tools, or expert creation criteria"
        }}
        """
        return reflection_prompt
