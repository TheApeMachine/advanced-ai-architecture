import torch
import torch.nn as nn
import torch.optim as optim

class MAMLModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MAMLModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

def maml_step(model, optimizer, tasks, inner_lr=0.01, meta_lr=0.001):
    meta_loss = 0
    for task in tasks:
        x_train, y_train, x_val, y_val = task
        # Clone model for inner loop
        inner_model = MAMLModel(input_size, hidden_size, output_size)
        inner_model.load_state_dict(model.state_dict())
        inner_optimizer = optim.SGD(inner_model.parameters(), lr=inner_lr)
        # Inner loop
        inner_optimizer.zero_grad()
        y_pred = inner_model(x_train)
        loss = nn.functional.cross_entropy(y_pred, y_train)
        loss.backward()
        inner_optimizer.step()
        # Meta-update
        y_pred = inner_model(x_val)
        meta_loss += nn.functional.cross_entropy(y_pred, y_val)
    meta_loss /= len(tasks)
    optimizer.zero_grad()
    meta_loss.backward()
    optimizer.step()
    return meta_loss.item()

# Example usage
input_size = 10
hidden_size = 40
output_size = 5

model = MAMLModel(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assume tasks is a list of tasks with training and validation data
for epoch in range(100):
    loss = maml_step(model, optimizer, tasks)
    print(f"Epoch {epoch}, Meta Loss: {loss}")
