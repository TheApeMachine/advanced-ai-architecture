from huggingface_hub import hf_hub_download

class Domain:
    def __init__(self):
        self.models = {
            "code": "codellama/CodeLlama-13b-Instruct-hf",
            "math": "Qwen/Qwen2-Math-7B-Instruct",
            "general": {
                "model_name": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                "model_file": "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
            }
        }
        
        self.tools = {
            "code": ["communicator", "assistant", "tasks"],
            "math": ["communicator", "assistant", "tasks"],
            "management": ["communicator", "assistant", "tasks"],
        }

    def equip(self, name):
        return {
            "model": hf_hub_download(self.map[name]["model_name"], filename=self.map[name]["model_file"]),
            "tools": self.tools[name]
        }
