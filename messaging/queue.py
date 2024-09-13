from logger import Logger
        
class Queue:
    """
    Queue is a system for managing messages between Expert agents.
    Messages can be sent to a single Expert agent, a group of Expert agents, or all Expert agents.
    Each Expert agent has an "inbox" which contains all the messages sent to them while they were
    not running. When an Expert agent runs, their inbox will be injected into their context and then
    cleared.
    
    While Assistant agents are also Expert agents, they are fully isolated, and only interact with
    the Expert agent that created them.
    """
    def __init__(self):
        self.logger = Logger(name='messageslogger')
        self.queues = {
            "broadcast": [],
            "direct": {},
            "rooms": {}
        }
        self.experts = {}
        self.channels = {}

    def register_expert(self, expert):
        """
        Register a new expert into the message queue system.
        :param expert: The Expert instance to register.
        """
        self.experts[expert.name] = expert
        self.queues["direct"][expert.name] = []
    
    def broadcast(self, message, sender):
        """
        Broadcast a message to all registered experts.
        :param message: The message to broadcast.
        :param sender: The sender of the message.
        """
        self.logger.info(f"{sender} broadcasting: {message}")
        self.queues["broadcast"].append((sender, message))
        for expert_name, expert in self.experts.items():
            if expert_name != sender:  # Avoid sending back to the sender
                expert.receive_message(f"Broadcast from {sender}: {message}")
    
    def send_direct_message(self, message, sender, recipient):
        """
        Send a direct message to a specific expert.
        :param message: The message to send.
        :param sender: The expert sending the message.
        :param recipient: The expert receiving the message.
        """
        if recipient in self.experts:
            self.logger.info(f"{sender} sending direct message to {recipient}: {message}")
            self.queues["direct"][recipient].append((sender, message))
            self.experts[recipient].receive_message(f"Direct message from {sender}: {message}")
        else:
            self.logger.error(f"Error: {recipient} not found in the system.")
    
    def create_room(self, room_name, participants):
        """
        Create a breakout room for a group of experts.
        :param room_name: The name of the room.
        :param participants: A list of expert names to include in the room.
        """
        self.queues["rooms"][room_name] = {
            "participants": participants,
            "messages": []
        }
        self.logger.info(f"Breakout room '{room_name}' created for {', '.join(participants)}.")
    
    def send_room_message(self, room_name, message, sender):
        """
        Send a message to all participants in a breakout room.
        :param room_name: The name of the room.
        :param message: The message to send.
        :param sender: The sender of the message.
        """
        if room_name in self.queues["rooms"]:
            participants = self.queues["rooms"][room_name]["participants"]
            self.logger.info(f"{sender} sending message to room {room_name}: {message}")
            self.queues["rooms"][room_name]["messages"].append((sender, message))
            for participant in participants:
                if participant != sender:
                    self.experts[participant].receive_message(f"Room {room_name} message from {sender}: {message}")
        else:
            self.logger.error(f"Error: Room {room_name} not found.")
    
    def retrieve_messages(self, expert_name):
        """
        Retrieve all messages (broadcast and direct) for a specific expert.
        :param expert_name: The expert for whom to retrieve messages.
        :return: List of messages.
        """
        if expert_name in self.experts:
            # Gather messages from all sources
            direct_messages = self.queues["direct"].get(expert_name, [])
            broadcast_messages = self.queues["broadcast"]
            
            return {
                "broadcast": broadcast_messages,
                "direct": direct_messages
            }
        else:
            return f"Error: {expert_name} not registered."

    def create_channel(self, name):
        if name not in self.channels:
            self.channels[name] = []

    def send_message(self, channel, sender, message):
        if channel not in self.channels:
            self.create_channel(channel)
        self.channels[channel].append({"sender": sender, "message": message})

    def get_messages(self, channel):
        return self.channels.get(channel, [])

