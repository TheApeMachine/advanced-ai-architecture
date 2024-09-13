from .tool import Tool

class Assistant(Tool):
    def __init__(self):
        super().__init__(
            name="assistant",
            description="Create a new assistant Expert.",
            parameters={
                "name": {
                    "type": "string",
                    "description": "The name of the assistant."
                },
                "domain": {
                    "type": "string",
                    "description": "The domain expertise of the assistant."
                }
            }
        )

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

    def use(self, expert_creator, **kwargs):
        # Instead of creating the Expert directly, we'll return the parameters
        # The Expert creation will be handled by the caller
        return {
            "action": "create_expert",
            "parameters": kwargs
        }