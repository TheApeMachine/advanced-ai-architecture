from typing import Dict, Any
from prompt import Prompt
from tools.toolset import Toolset
from messaging.queue import Queue
from collections import deque
from utils import process_streaming_response

class Expert:
    def __init__(self, name: str, domain: str, model, prompt: Prompt, toolset: Toolset, message_queue: Queue):
        self.name = name
        self.domain = domain
        self.model = model
        self.prompt = prompt
        self.toolset = toolset
        self.message_queue = message_queue
        self.context = {}
        self.assistants = {}
        self.task_queue = deque()

        # Add tools to the model
        for tool in self.toolset.get_tools():
            self.model.add_tool(tool.to_dict())
            
    async def run(self):
        """
        run iterates the Expert agent, processing tasks from the task queue and updating the context as it goes.
        """
        await process_streaming_response(
            self.model.generate(
                self.prompt.generate_prompt(self.domain, self.task_queue[0], self.context)
            )
        )
