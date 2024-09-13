class Task:
    """
    Task is a structured representation of a task.
    
    """
    
    def __init__(self, instructions, criteria):
        self.instructions = instructions
        self.criteria = criteria
        self.status = 0

    def __str__(self):
        return f"Task(instructions={self.instructions}, criteria={self.criteria})"
    
    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            "instructions": self.instructions,
            "criteria": self.criteria
        }

    def result(self, result):
        self.result = result
        self.status = self.workflow.index("completed")

    def is_complete(self):
        return self.status == self.workflow.index("completed")

    def is_failed(self):
        return self.status == self.workflow.index("failed")