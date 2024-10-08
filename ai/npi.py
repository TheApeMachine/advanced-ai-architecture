import torch
import torch.nn as nn

class NPI(nn.Module):
    def __init__(self, input_size, hidden_size, num_subroutines):
        super(NPI, self).__init__()
        self.hidden_size = hidden_size
        self.subroutine_embedding = nn.Embedding(num_subroutines, hidden_size)
        self.lstm = nn.LSTMCell(input_size + hidden_size, hidden_size)
        self.subroutine_selector = nn.Linear(hidden_size, num_subroutines)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, subroutine_id, h, c):
        subroutine_embed = self.subroutine_embedding(subroutine_id)
        lstm_input = torch.cat([x, subroutine_embed], dim=1)
        h, c = self.lstm(lstm_input, (h, c))
        next_subroutine_logits = self.subroutine_selector(h)
        output = self.output_layer(h)
        return output, next_subroutine_logits, h, c
