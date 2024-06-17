import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureGate(nn.Module):
    def __init__(self, cfg):
        super(FeatureGate, self).__init__()
        self.num_units = 400
     
        self.f_dense = nn.Linear(self.num_units * 2, self.num_units)  
        self.g_dense = nn.Linear(self.num_units * 2, self.num_units)

    def forward(self, short_rep, long_rep):
        # short_rep, long_rep: [batch_size, time, units] = [16, 50, 400]

        # Concatenation
        f_input = torch.cat([short_rep, long_rep], dim=-1)  
        g_input = torch.cat([short_rep, long_rep], dim=-1)

        # Transformation
        f = torch.tanh(self.f_dense(f_input))
        g = torch.sigmoid(self.g_dense(g_input))

        # Gating
        outputs = torch.mul(g, short_rep) + torch.mul(1 - g, f)

        return outputs

