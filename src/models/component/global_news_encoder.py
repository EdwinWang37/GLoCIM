import copy

import torch
import torch.nn as nn
import numpy as np
from src.models.base.layers import *
from torch_geometric.nn import Sequential, GCNConv
from pathlib import Path

import copy

import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch.nn import LSTM



class GlobalNewsEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.news_dim = cfg.model.head_num * cfg.model.head_dim
        attention_input_dim = self.news_dim




    def forward(self, x, edge_index, mask=None):



  

        result = self.last_encoder(last_word_emb)


        return result.view(batch_size, num_news, self.news_dim)
