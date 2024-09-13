class Channel:
    """
    Channel is a class that represents a channel that messages can be sent to.
    """

    def __init__(self, name):
        self.name = name
        self.messages = []

    def send(self, message):
        self.messages.append(message)

    def receive(self, message):
        self.messages.append(message)
