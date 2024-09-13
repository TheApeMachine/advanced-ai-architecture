import torch
import torch.nn as nn
import torch.optim as optim

class EWCModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EWCModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.fisher = {}
        self.optimal_params = {}

    def forward(self, x):
        return self.model(x)

    def compute_fisher(self, data_loader):
        # Initialize fisher information to zero
        for name, param in self.named_parameters():
            self.fisher[name] = torch.zeros_like(param)
        # Compute fisher information
        self.train()
        for x, y in data_loader:
            self.zero_grad()
            output = self.forward(x)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            for name, param in self.named_parameters():
                self.fisher[name] += param.grad.data ** 2
        # Average over dataset
        for name in self.fisher:
            self.fisher[name] /= len(data_loader)

    def consolidate(self):
        # Save optimal parameters
        for name, param in self.named_parameters():
            self.optimal_params[name] = param.data.clone()

    def ewc_loss(self, loss):
        if not self.fisher:
            return loss
        ewc = 0
        for name, param in self.named_parameters():
            ewc += (self.fisher[name] * (param - self.optimal_params[name]) ** 2).sum()
        return loss + 0.5 * ewc
