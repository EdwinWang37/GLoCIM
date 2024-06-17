import copy

import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential, GCNConv
from pathlib import Path

import copy

import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *


class NewsEncoder(nn.Module):
    def __init__(self, cfg, glove_emb=None):
        super().__init__()
        token_emb_dim = cfg.model.word_emb_dim
        self.news_dim = cfg.model.head_num * cfg.model.head_dim

        if cfg.dataset.dataset_lang == 'english':
            pretrain = torch.from_numpy(glove_emb).float()
            self.word_encoder = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)
        else:
            self.word_encoder = nn.Embedding(glove_emb+1, 300, padding_idx=0)
            nn.init.uniform_(self.word_encoder.weight, -1.0, 1.0)

        self.view_size = [cfg.model.title_size, cfg.model.abstract_size]

  
        attention_input_dim = self.news_dim + cfg.model.category_emb_dim + cfg.model.subcategory_emb_dim


        self.attention = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            (MultiHeadAttention(attention_input_dim,
                                attention_input_dim,
                                attention_input_dim,
                                cfg.model.head_num,
                                50), 'x,x,x,mask -> x'),
            nn.LayerNorm(attention_input_dim),
            nn.Dropout(p=cfg.dropout_probability),

            (AttentionPooling(attention_input_dim,
                              cfg.model.attention_hidden_dim), 'x,mask -> x'),
            nn.LayerNorm(attention_input_dim),
        ])
        self.last_encoder = nn.Linear(1000, 400)

        self.attetio = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            (MultiHeadAttention(token_emb_dim,
                                token_emb_dim,
                                token_emb_dim,
                                cfg.model.head_num,
                                cfg.model.head_dim), 'x,x,x,mask -> x'),  # 20 * 20
            nn.LayerNorm(self.news_dim),  # 400
            nn.Dropout(p=cfg.dropout_probability),  # 0.2

            (AttentionPooling(self.news_dim,
                              cfg.model.attention_hidden_dim), 'x,mask -> x'),
            nn.LayerNorm(self.news_dim),  # 400
            # nn.Linear(self.news_dim, self.news_dim),
            # nn.LeakyReLU(0.2),
        ])


    def forward(self, news_input, mask=None):
        """
                Args:
                    news_input:  [batch_size, news_num, total_input]
                    mask:   [batch_size, news_num]
                Returns:
                    [batch_size, news_num, news_emb] eg. [64,50,400]
        """

        batch_size = news_input.shape[0]

        num_news = news_input.shape[1]
  

        title_input, _ ,category_input ,subcategory_input, _ = news_input.split([self.view_size[0], 5, 1, 1, 1], dim=-1)




        title_word_emb = self.word_encoder(title_input.long().view(-1, self.view_size[0]))
        category_emb = self.word_encoder(category_input.long().view(-1))
        subcategory_emb = self.word_encoder(subcategory_input.long().view(-1))


        total_word_emb = self.attetio(title_word_emb, mask)



        fuse_word_emb = torch.cat([total_word_emb, category_emb, subcategory_emb], dim=1)



        last_word_emb = self.attention(fuse_word_emb, mask)



        result = self.last_encoder(last_word_emb)




        return result.view(batch_size, num_news, self.news_dim)
