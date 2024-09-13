import asyncio
from llama_cpp import Llama
import json

class Model:
    """
    Model is a wrapper around a machine learning model.
    It provides a consistent interface for the rest of the application to use.
    """
    
    def __init__(self, model_path):
        """
        Initialize the model with the given model path.
        """
        self.model = Llama(
            model_path=model_path, 
            n_ctx=2048, 
            n_gpu_layers=-1
        )
        
        self.characters = {
            "default": {"max_new_tokens": 1024, "temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.1},
            "creative": {"max_new_tokens": 1024, "temperature": 1.2, "top_p": 0.95, "repeat_penalty": 1.1},
            "precise": {"max_new_tokens": 1024, "temperature": 0.5, "top_p": 0.9, "repeat_penalty": 1.1},
            "strict": {"max_new_tokens": 1024, "temperature": 0.0, "top_p": 0.9, "repeat_penalty": 1.1},
        }

    async def generate(self, prompt, max_new_tokens=1024, temperature=0.7, top_p=0.9, repeat_penalty=1.1):
        """
        Generate a response from the model.
        Experts are able to tweak themselves to optimize their performance.
        The following parameters are available to tweak the performance:
        - max_new_tokens: The maximum number of tokens to generate.
        - temperature: controls the randomness or creativity of the model.
        - top_p: controls the diversity of the model, which is the probability that the model will choose a less probable token.
        - repeat_penalty: controls the likelihood of the model repeating itself.
        """

        async def stream_response():
            for output in self.model.create_completion(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stream=True
            ):
                chunk = output['choices'][0]['text']

        return stream_response()
    