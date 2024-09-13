import json
from termcolor import colored

def format_partial_json(key, value):
    """
    Colorize individual parts of the JSON based on the key for user-friendly display.
    """
    if key == "thought_process":
        return f"{colored('Thought:', 'cyan')} {value}\n"
    elif key == "decision":
        return f"{colored('Decision:', 'green')} {value}\n"
    elif key == "action":
        return f"{colored('Action:', 'yellow')} {value}\n"
    elif key == "action_details":
        return f"{colored('Action Details:', 'magenta')} {json.dumps(value, indent=2)}\n"
    elif key == "next_steps":
        return f"{colored('Next Steps:', 'blue')} {', '.join(value)}\n"
    else:
        return f"{key}: {value}\n"

async def process_streaming_response(response_generator):
    """
    Handle both system and user-friendly streaming responses.
    
    Args:
        response_generator: The async generator providing the streamed response chunks.
        process_for_system: Function to handle full response for system (buffering JSON).
        stream_to_user: Function to handle user-friendly output (ignoring JSON structure).
    """
    internal_buffer = ""  # Buffer for internal processing
    buffer = ""  # Buffer for incomplete JSON chunks

    async for chunk in response_generator:
        internal_buffer += chunk  # Accumulate full response for system use
        buffer = process_streaming_json_chunk(chunk, buffer, stream_to_user)

    # When complete, handle the full response for system processing
    await process_for_system(internal_buffer)

def process_streaming_json_chunk(chunk, buffer, stream_to_user):
    """
    Process each chunk of JSON as it arrives, handling incomplete JSON objects and streaming output.
    
    Args:
        chunk: The current chunk of data to process.
        buffer: The current buffer that holds incomplete JSON.
        stream_to_user: Function to stream data for user-friendly display.
    
    Returns:
        Updated buffer with remaining incomplete data.
    """
    buffer += chunk  # Append new chunk to buffer
    try:
        while buffer:
            # Attempt to load any complete JSON objects from the buffer
            data, index = json.JSONDecoder().raw_decode(buffer)
            buffer = buffer[index:].lstrip()  # Remove parsed data and continue with leftover buffer

            # Stream the JSON data in a user-friendly format
            for key, value in data.items():
                stream_to_user(key, value)
    except json.JSONDecodeError:
        # If we don't have a full JSON object yet, return and wait for more data
        pass

    return buffer

def user_friendly_output(chunk):
    """
    Display user-friendly output by stripping out JSON structure and showing only relevant values.
    
    Args:
        chunk: The current chunk of text to process and display.
    """
    value = ""
    in_quotes = False

    for char in chunk:
        if char == '"':
            in_quotes = not in_quotes  # Toggle the state inside/outside of quotes
        elif in_quotes:
            value += char  # Capture the actual content between quotes
        elif char == ':':
            value += ": "
        elif char in [',', '{', '}']:
            if value.strip():
                print(colored(value, "green"), end='', flush=True)
                value = ""
            print(" ", end='', flush=True)

    if value.strip():
        print(colored(value, "green"), end='', flush=True)

async def process_full_response_for_system(response):
    """
    Handle the full response for system use, e.g., interpreting actions or making decisions.
    """
    try:
        action = json.loads(response)
        # Call the system's internal logic to process the action, e.g., execute a task
        # For example, this could be expert.interpret_response(action) if within Expert class
        print(f"System handling action: {action}")
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from response: {response}")
