from .tool import Tool

class Tasks(Tool):
    def __init__(self):
        super().__init__(
            name="tasks",
            description="Schedule multiple tasks onto a queue, to manage context length.",
            parameters={
                "tasklist": {
                    "type": "string",
                    "description": "A task to be scheduled onto the queue formatted as a prompt."
                }
            }
        )

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

    async def use(self, **kwargs):
        """
        Schedule a task onto the queue.
        """
        pass