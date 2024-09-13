from huggingface_hub import hf_hub_download

class Domain:
    def __init__(self):
        self.models = {
            "code": {
                "model_name": "codellama/CodeLlama-13b-Instruct-hf",
                "model_file": "pytorch_model.bin"  # You may need to adjust this filename
            },
            "math": {
                "model_name": "Qwen/Qwen2-Math-7B-Instruct",
                "model_file": "pytorch_model.bin"  # You may need to adjust this filename
            },
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
        model_info = self.models[name]
        return {
            "model": hf_hub_download(
                repo_id=model_info["model_name"],
                filename=model_info["model_file"]
            ),
            "tools": self.tools.get(name, [])
        }
