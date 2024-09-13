import torch
import torch.nn as nn

class HyperNetwork(nn.Module):
    def __init__(self, z_dim, target_dims):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.target_dims = target_dims
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, sum([torch.prod(torch.tensor(dim)) for dim in target_dims]))
        )

    def forward(self, z):
        weights = self.fc(z)
        return self.construct_weights(weights)

    def construct_weights(self, flat_weights):
        weights = []
        offset = 0
        for dim in self.target_dims:
            param_size = torch.prod(torch.tensor(dim))
            param_shape = dim
            param_weights = flat_weights[:, offset:offset + param_size].view(-1, *param_shape)
            weights.append(param_weights)
            offset += param_size
        return weights

class TargetNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TargetNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, x, weights):
        w1, b1, w2, b2 = weights
        x = torch.relu(torch.matmul(x, w1.t()) + b1)
        x = torch.matmul(x, w2.t()) + b2
        return x

# Usage
input_size = 10
hidden_size = 20
output_size = 1

# Define target network dimensions
target_dims = [(hidden_size, input_size),  # w1
               (hidden_size,),             # b1
               (output_size, hidden_size), # w2
               (output_size,)]             # b2

hyper_net = HyperNetwork(z_dim=5, target_dims=target_dims)
target_net = TargetNetwork(input_size, hidden_size, output_size)

# Task-specific embedding
z = torch.randn(1, 5)
weights = hyper_net(z)

# Input data
x = torch.randn(1, input_size)
output = target_net(x, weights)
print(output)
