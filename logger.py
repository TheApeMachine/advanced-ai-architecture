import logging
import sys
from coloredlogs import install, ColoredFormatter

class CircularBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def append(self, item):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(item)
        
    def get_last_messages(self):
        return self.buffer
    
    def get_last_message(self):
        # Only return the last message, for checking short sequences.
        return self.buffer[-1]

class Logger:
    """
    Logger is a class that represents the output meant for the user.
    It should be focused on being human-readable, easily understandable, and concise.
    It should also be used to show the streaming responses of the AI models.
    """
    
    def __init__(self, name):
        logging.basicConfig()
        self.logger = logging.getLogger(name=name)
        install(logger=self.logger)
        self.logger.propagate = False

        self.coloredFormatter = ColoredFormatter(
            fmt='[%(name)s] %(asctime)s %(funcName)s %(lineno)-3d  %(message)s',
            level_styles=dict(
                debug=dict(color='white'),
                info=dict(color='blue'),
                warning=dict(color='yellow', bright=True),
                error=dict(color='red', bold=True, bright=True),
                critical=dict(color='black', bold=True, background='red'),
            ),
            field_styles=dict(
                name=dict(color='white'),
                asctime=dict(color='white'),
                funcName=dict(color='white'),
                lineno=dict(color='white'),
            )
        )

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(fmt=self.coloredFormatter)
        self.logger.addHandler(hdlr=ch)
        self.logger.setLevel(level=logging.DEBUG)
        
        self.internal_buffer = ""
        
        self.structural_tokens = ['{', '}', '[', ']', ',', ':', '"', "'"]
        self.sequences = {
            "key": ['"'], # If we see this sequence, and we are not in a value, we know we are streaming a key.
            "value": ['"', ":", '"'], # If we see this sequence, we know we are streaming a value.
            "end": ['"'] # If we see this sequence, and we are in a value, know we are at the end of a JSON object.
        }
        
        self.in_key = False
        self.in_value = False
        
        self.key = None
        self.key_colors = {
            "name": "white",
            "model": "green",
            "prompt_type": "magenta",
            "response": "white"
        }
        
        # We need a circular buffer with 3 slots to store the last 3 messages, which is enough to
        # evaluate if we have seen the value sequence.
        self.circular_buffer = CircularBuffer(capacity=3)
        
    async def response(self, generator):
        """
        response can plug in between the LLM and the internal buffer to provide a user-friendly output.
        It should ignore any structure elements of the response, recognize when it has seen a key,
        and only show the value of the key. The key itself is only used to select a color for the value.
        No matter what happens during the human-readable output, the LLM will still receive the full
        response so that it can continue to stream the response as a system message.
        """
        self.internal_buffer = ""
        async for chunk in generator:
            # We add the chunk to the internal buffer first, so we prioritize the system-level responses.
            self.internal_buffer += chunk
            
            # We then see if we are currently streaming structural elements, such as { or [, and
            # if we are, we display a spinner to the user to show that we are still processing.
            for piece in chunk.split():
                if piece in self.structural_tokens:
                    self.circular_buffer.append(piece)
            
                # We then check if we have seen the key sequence.
                if self.circular_buffer.get_last_message() == self.sequences["key"]:
                    self.in_key = True
                
                # We check for the end of the key.
                if self.in_key and piece == self.sequences["key"]:
                    self.in_key = False
                
                # We add the piece to the key if we are still in the key.
                if self.in_key:
                    self.key += piece
                
                # We then check if we have seen the value sequence.
                if self.circular_buffer.get_last_messages() == self.sequences["value"]:
                    self.in_value = True
                
                # We check for the end of the value.
                if self.in_value and piece == self.sequences["end"]:
                    self.in_value = False
                
                # We add the piece to the value if we are still in the value.
                if self.in_value:
                    self.logger.log(level=self.key_colors[self.key], msg=piece)

        return self.internal_buffer

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
