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

# Example usage
input_size = 10
hidden_size = 64
num_classes = 5

model = ProtoNet(input_size, hidden_size)

# Assume support_set and query_set are provided
support_embeddings = model(support_set['x'])
prototypes = compute_prototypes(support_embeddings, support_set['y'], num_classes)

query_embeddings = model(query_set['x'])
distances = torch.cdist(query_embeddings, prototypes)
log_p_y = F.log_softmax(-distances, dim=1)
loss = F.nll_loss(log_p_y, query_set['y'])
