import torch
import torch.nn as nn

class NCPCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NCPCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_weights = nn.Linear(input_size, hidden_size)
        self.recurrent_weights = nn.Linear(hidden_size, hidden_size)
        self.output_weights = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()

    def forward(self, x, h_prev):
        h = self.activation(self.input_weights(x) + self.recurrent_weights(h_prev))
        y = self.output_weights(h)
        return y, h

# Example usage
input_size = 10
hidden_size = 50

ncp_cell = NCPCell(input_size, hidden_size)
h_prev = torch.zeros(1, hidden_size)
x = torch.randn(1, input_size)

y, h_prev = ncp_cell(x, h_prev)
print(y)
