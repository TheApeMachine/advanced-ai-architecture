import torch
import torch.nn as nn
import torch.optim as optim

class LiquidTimeConstantLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidTimeConstantLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        # Time constants for each neuron
        self.tau = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.uniform_(self.tau, 0.1, 1.0)  # Initialize time constants

    def forward(self, x, h_prev):
        tau = torch.abs(self.tau)  # Ensure time constants are positive
        # Update rule incorporating the time constant
        h = (1 - 1 / tau) * h_prev + (1 / tau) * self.rnn_cell(x, h_prev)
        return h

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LiquidNeuralNetwork, self).__init__()
        self.liquid_layer = LiquidTimeConstantLayer(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_prev):
        h = self.liquid_layer(x, h_prev)
        y = self.output_layer(h)
        return y, h

# Example usage
input_size = 10
hidden_size = 50
output_size = 10

model = LiquidNeuralNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Initialize hidden state
h_prev = torch.zeros(1, hidden_size)

# Simulate streaming data
for t in range(1000):  # Time steps
    x = torch.randn(1, input_size)  # New input at each time step
    target = x.sum(dim=1, keepdim=True)  # Example target
    y, h_prev = model(x, h_prev)
    loss = criterion(y, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
