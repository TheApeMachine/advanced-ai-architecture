class Message:
    """
    Message is a class that represents a message that can be sent between Expert agents.
    """

    def __init__(self, channel, sender, subject, message):
        self.channel = channel
        self.sender = sender
        self.subject = subject
        self.message = message
        
    def __str__(self):
        return f"""
        <MESSAGE>
        FROM: {self.sender}
        TO: {self.channel}
        SUBJECT: {self.subject}
        
        {self.message}
        </MESSAGE>
        """