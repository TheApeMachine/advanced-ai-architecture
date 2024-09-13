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

# Usage
input_size = 784  # For MNIST
hidden_size = 256
output_size = 10

model = EWCModel(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Assume task1_loader and task2_loader are DataLoader objects for two tasks
# Training on Task 1
for epoch in range(5):
    for x, y in task1_loader:
        optimizer.zero_grad()
        output = model(x.view(-1, 784))
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
model.compute_fisher(task1_loader)
model.consolidate()

# Training on Task 2 with EWC
for epoch in range(5):
    for x, y in task2_loader:
        optimizer.zero_grad()
        output = model(x.view(-1, 784))
        loss = nn.functional.cross_entropy(output, y)
        total_loss = model.ewc_loss(loss)
        total_loss.backward()
        optimizer.step()
