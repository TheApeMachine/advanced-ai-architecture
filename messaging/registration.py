class Registration:
    """
    Registration is an object that represents a mapping of Expert agents to their channels.
    """

    def __init__(self):
        self.inbox = []
        self.contacts = []
        self.channels = []


    def register(self, expert, channel):
        if channel not in self.channels:
            self.channels[channel] = []
        self.channels[channel].append(expert)
        
