from transformers import BertTokenizer, BertModel
from .nas_nn import NASNeuralNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit_algorithms import QAOA
from qiskit_optimization import QuadraticProgram

class Coder:
    def __init__(self):
        self.code = ""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.model = NASNeuralNetwork()
        self.linear_projection = nn.Linear(768, 10)

    def generate(self, description):
        # Use BERT to process the description
        encoded_input = self.tokenizer(description, return_tensors='pt')
        bert_output = self.bert_model(**encoded_input)
        
        # Average over the sequence dimension to get a [batch_size, hidden_size] tensor
        code_representation = torch.mean(bert_output.last_hidden_state, dim=1)  # Now [1, 768]

        # Optionally project the 768-dimensional BERT output to 10 dimensions
        # Uncomment if needed:
        # code_representation = self.linear_projection(code_representation)  # Now [1, 10] if used

        # Pass the representation to NASNeuralNetwork, which dynamically handles input size
        structured_code = self.model(code_representation)

        # Simplified decoding for now
        generated_code = self.decode_to_code(structured_code)
        
        return generated_code
    
    def create_code_structure_graph(self, code_representation):
        # Simplified graph creation based on code representation similarities
        num_nodes = code_representation.size(0)
        edge_index = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if F.cosine_similarity(code_representation[i], code_representation[j], dim=0) > 0.5:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def optimize_code_structure(self, structured_code):
        # Define a QuadraticProgram for code structure optimization
        qp = QuadraticProgram()
        for i in range(structured_code.size(0)):
            qp.binary_var(name=f'node_{i}')
        
        # Define the objective function (example: maximize node importance)
        node_importance = structured_code.sum(dim=1)
        linear = {f'node_{i}': float(imp) for i, imp in enumerate(node_importance)}
        qp.maximize(linear=linear)
        
        # Add constraints (example: limit total nodes selected)
        qp.linear_constraint(linear={f'node_{i}': 1 for i in range(structured_code.size(0))}, 
                             sense='LE', rhs=structured_code.size(0)//2, name='max_nodes')
        
        # Run QAOA
        qaoa = QAOA(optimizer=self.optimizer, sampler=self.sampler, reps=1)
        result = qaoa.compute_minimum_eigenvalue(qp)
        
        # Apply the optimization result to the structured code
        optimized_code = structured_code * torch.tensor([result.x[f'node_{i}'] for i in range(structured_code.size(0))])
        return optimized_code

    def decode_to_code(self, code_representation):
        # For now, we're keeping this simple by generating basic Python functions
        # We can make this more advanced by using sequence models or rule-based decoders
        python_code = "def add_two_numbers(a, b):\n"
        python_code += "    return a + b\n"
    
        # Later, we can dynamically generate function names and content based on the task
        return python_code