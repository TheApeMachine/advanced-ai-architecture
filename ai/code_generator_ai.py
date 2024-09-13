from transformers import AutoTokenizer, AutoModelForCausalLM
from .context_manager import ContextManager
import torch
from transformers import BertModel, BertTokenizer
from .multi_modal_encoders import PseudocodeEncoder, DiagramEncoder
from world.domain import Domain
from .model import Model

class CodeGeneratorAI:
    def __init__(self, model_path):
        self.model = Model(model_path)
        self.reward_history = []
        self.temperature = 0.7
        self.previous_params = {}
        self.fisher_information = {}
        self.context_manager = ContextManager()
        self.style_embeddings = self.load_style_embeddings()
        self.style = 'default'
        self.nl_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.pseudocode_encoder = PseudocodeEncoder()
        self.diagram_encoder = DiagramEncoder()

    def load_style_embeddings(self):
        # Load pre-defined style embeddings
        # Placeholder implementation
        return {'default': torch.zeros(768), 'pep8': torch.randn(768), 'google_style': torch.randn(768)}

    def set_style(self, style_name):
        if style_name in self.style_embeddings:
            self.style = style_name
        else:
            print(f"Style {style_name} not found. Using default style.")

    async def generate_code(self, prompt, pseudocode=None, diagram=None, max_length=1024):
        full_prompt = self._prepare_prompt(prompt, pseudocode, diagram)
        
        async for chunk in self.model.generate(
            full_prompt,
            max_new_tokens=max_length,
            temperature=self.model.characters["default"]["temperature"],
            top_p=self.model.characters["default"]["top_p"],
            repeat_penalty=self.model.characters["default"]["repeat_penalty"]
        ):
            yield chunk

    def _prepare_prompt(self, prompt, pseudocode=None, diagram=None):
        full_prompt = f"### Instruction: Generate code based on the following prompt:\n{prompt}\n"
        
        if pseudocode:
            full_prompt += f"\nPseudocode:\n{pseudocode}\n"
        
        if diagram:
            full_prompt += f"\nDiagram description:\n{diagram}\n"
        
        full_prompt += "\n### Response:"
        return full_prompt
    
    async def generate_code_batch(self, prompts, max_length=1024):
        codes = []
        for prompt in prompts:
            full_code = ""
            async for chunk in self.generate_code(prompt, max_length=max_length):
                full_code += chunk
            codes.append(full_code)
        return codes

    def adjust_parameters(self, reward):
        self.reward_history.append(reward)
        if reward > 0:
            self.temperature = max(self.temperature - 0.01, 0.5)
        else:
            self.temperature = min(self.temperature + 0.01, 1.0)
            
    def consolidate_weights(self):
        # Calculate Fisher Information matrix and store parameters
        for name, param in self.model.named_parameters():
            self.previous_params[name] = param.clone()
            self.fisher_information[name] = self.compute_fisher_information(param)

    def compute_fisher_information(self, param):
        # Implement computation of Fisher Information
        # Placeholder implementation
        return torch.ones_like(param)

    def ewc_loss(self):
        loss = 0
        for name, param in self.model.named_parameters():
            loss += torch.sum(self.fisher_information[name] * (param - self.previous_params[name]) ** 2)
        return loss

    def train(self, data_loader):
        # Training loop
        for data in data_loader:
            output = self.model(data)
            loss = self.compute_loss(output, data['target']) + self.ewc_loss()
            # Backpropagation and optimization
