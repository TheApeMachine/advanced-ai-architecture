class Tool:
    """
    Tool wraps additional functionality that can be used to help the Expert.
    All speciliazed behavior should be wrapped in a Tool.
    """
    
    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

    def use(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
