from tools.assistant import Assistant
from tools.tasks import TaskBreakdown

class Toolset:
    def __init__(self, tools=[]):
        self.tools = {}
        for tool in tools:
            self.add_tool(tool)
        self.add_tool('task_breakdown')  # Always add the TaskBreakdown tool

    def add_tool(self, tool_name):
        if tool_name == 'assistant':
            self.tools[tool_name] = Assistant()
        elif tool_name == 'task_breakdown':
            self.tools[tool_name] = TaskBreakdown()
        # Add other tools as needed

    def get_tool(self, tool_name):
        return self.tools.get(tool_name)

    def get_tools(self):
        return list(self.tools.values())

    def get_tool_names(self):
        return list(self.tools.keys())