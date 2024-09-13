from .tool import Tool

class Communicator(Tool):
    """
    Communicator is a tool that allows the Expert agent to use the message system to communicate
    with other Expert agents. There are a number of different types of message channels, such as:
    - direct_message: a message that is sent to a single Expert agent, and any Assistant agents.
    - group_message: a message that can be used if the Expert agent is part of a group.
    - broadcast_message: a message that is sent to all Expert agents, though not to Assistants.
    """
    
    def __init__(self, messages):
        self.messages = messages

    def to_dict(self):
        return {
            "name": "communicator",
            "description": "A tool for communicating with the user."
        }
        
    def use(self, channel_type, channel_name, message):
        """
        Use the communicator tool to send a message to a channel.
        """
        if self.messages is None:
            raise ValueError("Messages system not available.")
        
        self.messages.add_message(channel_type, channel_name, message)
        