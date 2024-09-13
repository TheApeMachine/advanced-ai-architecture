class Toolset:
    """
    Toolset is a management interface for a set of tools.
    The Expert should use this over interacting with the tools directly.
    Generally, the tools in the toolset are determined by the Domain.
    """
        
    def __init__(self, tools=[]):
        self.tools = {}
        for tool in tools:
            self.add_tool(tool)
            
    def __str__(self):
        return f"Toolset({self.tools})"
    
    def to_dict(self):
        return self.tools

    def add_tool(self, tool):
        self.tools[tool.name] = tool

    def use(self, tool_name, **kwargs):
        if tool_name in self.tools:
            return self.tools[tool_name].use(**kwargs)
        else:
            raise ValueError(f"Tool {tool_name} not found")