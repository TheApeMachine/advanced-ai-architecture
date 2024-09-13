import torch.nn as nn

class NASNeuralNetwork(nn.Module):
    def __init__(self, hidden_sizes=None, output_size=10):
        super(NASNeuralNetwork, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128]  # Default architecture

        self.layers = nn.ModuleList()
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

    def forward(self, x, y=None):  # Add y=None to handle Pyro's additional argument
        # Get the input size dynamically
        in_size = x.shape[-1]  # Take the last dimension as input size
        
        # Build the layers dynamically based on the input size
        layers = []
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, self.output_size))
        self.model = nn.Sequential(*layers)

        return self.model(x)  # Only x is used for the forward pass
