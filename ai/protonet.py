import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.encoder(x)

def compute_prototypes(embeddings, labels, num_classes):
    prototypes = []
    for c in range(num_classes):
        class_embeddings = embeddings[labels == c]
        prototype = class_embeddings.mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes)
