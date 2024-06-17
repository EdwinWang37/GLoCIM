import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GATConv, GatedGraphConv


from models.component.candidate_encoder import *
from models.component.click_encoder import ClickEncoder
from models.component.entity_encoder import EntityEncoder, GlobalEntityEncoder
from models.component.nce_loss import NCELoss
from models.component.news_encoder import *
from models.component.user_encoder import *

from models.component.global_news_long_encoder import GlobalNewsLongEncoder

from models.component.feature_gate import FeatureGate
from utils.common import *

class LSP-GNR(nn.Module):
    def __init__(self, cfg, glove_emb=None, entity_emb=None):
        super().__init__()

        self.cfg = cfg
        self.use_entity = cfg.model.use_entity
        attention_input_dim = self.cfg.model.head_num * self.cfg.model.head_dim
        self.news_dim =  cfg.model.head_num * cfg.model.head_dim # 20 * 20 = 400
        self.entity_dim = cfg.model.entity_emb_dim #100

        # -------------------------- Model --------------------------
        # News Encoder
        self.local_news_encoder = NewsEncoder(cfg, glove_emb)

        #GCN
        self.global_news_short_encoder = Sequential('x, index', [
            (GatedGraphConv(self.news_dim, num_layers=3, aggr='add'),'x, index -> x'),
        ])

        #GAT
        # self.global_news_short_encoder = Sequential('x, edge_index', [
        #     (GATConv(self.news_dim, self.news_dim), 'x, edge_index -> x'),
        #     (GATConv(self.news_dim, self.news_dim), 'x, edge_index -> x'),
        #     (GATConv(self.news_dim, self.news_dim), 'x, edge_index -> x')
        # ])

        self.global_news_long_encoder = GlobalNewsLongEncoder(cfg)
        # self.global_news_long_encoder = Sequential('x, mask', [
        #     (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
        #     (MultiHeadAttention(attention_input_dim,
        #                         attention_input_dim,
        #                         attention_input_dim,
        #                         cfg.model.head_num,
        #                         cfg.model.head_dim), 'x,x,x,mask -> x'),
        #     nn.LayerNorm(attention_input_dim),
        #     nn.Dropout(p=cfg.dropout_probability),
        #
        #     (AttentionPooling(attention_input_dim,
        #                       cfg.model.attention_hidden_dim), 'x,mask -> x'),
        #     nn.LayerNorm(attention_input_dim),
        # ])


        # Entity
        if self.use_entity:
            pretrain = torch.from_numpy(entity_emb).float()
            self.entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)

            self.local_entity_encoder = Sequential('x, mask', [
                (self.entity_embedding_layer, 'x -> x'),
                (EntityEncoder(cfg), 'x, mask -> x'),
            ])

            self.global_entity_encoder = Sequential('x, mask', [
                (self.entity_embedding_layer, 'x -> x'),
                (GlobalEntityEncoder(cfg), 'x, mask -> x'),
            ])

        self.feature_gate = FeatureGate(cfg)
        # Click Encoder
        self.click_encoder = ClickEncoder(cfg)

        # User Encoder
        self.user_encoder = UserEncoder(cfg)
        
        # Candidate Encoder
        self.candidate_encoder = CandidateEncoder(cfg)

        # click prediction
        self.click_predictor = DotProduct()
        self.loss_fn = NCELoss()


    def forward(self, subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask,outputs_dict,trimmed_news_neighbors_dict,label=None):
        '''
        candidate_news: [32, 5, 38]
        candidate_entity:[32,5,55]   ----------------      ？？？
        entity_mask:[32,5,50]
        labels:[32,1]
        mapping_idx : [32, 50]
        outputs_dict : [51283,3,400]
        subgraph : 32
        trimmed_news_neighbors_dict：dict(51283 *(0-3))

        '''

        # -------------------------------------- clicked ----------------------------------

        mask = mapping_idx != -1
        mapping_idx[mapping_idx == -1] = 0




        batch_size, num_clicked, token_dim = mapping_idx.shape[0], mapping_idx.shape[1], candidate_news.shape[-1]
        clicked_entity = subgraph.x[mapping_idx, -8:-3]  #[16, 50, 5]
        click_history = subgraph.x[mapping_idx, -1] #[16, 50]


        x_flatten = subgraph.x.view(1, -1, token_dim)
        x_encoded = self.local_news_encoder(x_flatten).view(-1, self.news_dim)#[batch, news, news_dim] 32, 50, 400 -> 32 * 50, 400


        #neighbor
        global_short_emb = self.global_news_short_encoder(x_encoded, subgraph.edge_index)

        clicked_origin_emb = x_encoded[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked, self.news_dim)


        clicked_short_graph_emb = global_short_emb[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked, self.news_dim)

        # long-chain

        global_long_emb  = self.global_news_long_encoder(clicked_origin_emb, click_history, outputs_dict, trimmed_news_neighbors_dict)  # [16,50,400] 





        click_fuse_vec = self.feature_gate(clicked_short_graph_emb, global_long_emb)  #[16, 50, 400]

        # -------------ablation1-----------------------------
        #click_fuse_vec = clicked_short_graph_emb + global_long_emb
        #---------------------------------------------------


        # Attention pooling
        if self.use_entity:
            clicked_entity = self.local_entity_encoder(clicked_entity, None)
        else:
            clicked_entity = None

        clicked_total_emb = self.click_encoder(clicked_origin_emb, click_fuse_vec, clicked_entity)#local+global
        user_emb = self.user_encoder(clicked_total_emb, mask)
        # ----------------------------------------- Candidate------------------------------------
        cand_title_emb = self.local_news_encoder(candidate_news)                                      # [8, 5, 400]
        if self.use_entity:
            origin_entity, neighbor_entity = candidate_entity.split([self.cfg.model.entity_size,  self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)

            cand_origin_entity_emb = self.local_entity_encoder(origin_entity, None)
            cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)

        else:
            cand_origin_entity_emb, cand_neighbor_entity_emb = None, None



        cand_final_emb = self.candidate_encoder(cand_title_emb, cand_origin_entity_emb, cand_neighbor_entity_emb)
        # ----------------------------------------- Score ------------------------------------
        score = self.click_predictor(cand_final_emb, user_emb)
        loss = self.loss_fn(score, label)
        #print_model_memory_usage(self.global_news_short_encoder)
        #print_model_memory_usage(self.global_news_long_encoder)
        return loss, score

    def validation_process(self, subgraph, mappings, clicked_entity, candidate_emb, candidate_entity, entity_mask, outputs_dict, trimmed_news_neighbors_dict,click_history):
        
        batch_size, num_news, news_dim = 1 , len(mappings) , candidate_emb.shape[-1] #mappings



        # -----------------------------------------User-----------------------------------------------


        title_graph_emb = self.global_news_short_encoder(subgraph.x, subgraph.edge_index)

        clicked_origin_emb = subgraph.x[mappings, :].view(batch_size, num_news, news_dim)
        #1，15，400 

        click_history = torch.tensor(click_history)


        click_history = click_history.unsqueeze(0) #（1，50）
        ##print("click_history is：{}".format(click_history.shape))
        #print("click_history is：{}".format(click_history))

        #print("candidate_emb d:{}".format(candidate_emb.shape))#22,400
        #clicked_origin_emb = subgraph.x[mappings, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked, self.news_dim)

        global_long_emb = self.global_news_long_encoder(clicked_origin_emb, click_history, outputs_dict,
                                                        trimmed_news_neighbors_dict) # 1,50,400???

        truncated_emb = global_long_emb[:, :num_news, :]
       
   

        #print("title_graph_emb[mappings, :] is{}".format(title_graph_emb[mappings, :].shape))
        clicked_short_graph_emb = title_graph_emb[mappings, :].view(batch_size, num_news, news_dim) #【1，15，400】
        #print("clicked_short_graph_emb is{}".format(clicked_short_graph_emb))
        #print("clicked_short_graph_emb is{}".format(clicked_short_graph_emb.shape))
        #print("global_long_emb is{}".format(global_long_emb))
        #print("global_long_emb is{}".format(global_long_emb.shape))



        click_fuse_vec = self.feature_gate(clicked_short_graph_emb, truncated_emb) #1，15，400 和 1,50,400
        #------------------ab1-----------
        #click_fuse_vec = clicked_short_graph_emb + truncated_emb
        #-----------------------------------------------------------



        # --------------------Attention Pooling
        if self.use_entity:
            clicked_entity_emb = self.local_entity_encoder(clicked_entity.unsqueeze(0), None)
        else:
            clicked_entity_emb = None

        clicked_final_emb = self.click_encoder(clicked_origin_emb, click_fuse_vec, clicked_entity_emb)

        user_emb = self.user_encoder(clicked_final_emb)  # [1, 400]
        # ----------------------------------------- Candidate------------------------------------

        if self.use_entity:
            cand_entity_input = candidate_entity.unsqueeze(0)
            entity_mask = entity_mask.unsqueeze(0)
            origin_entity, neighbor_entity = cand_entity_input.split(
                [self.cfg.model.entity_size, self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)

            cand_origin_entity_emb = self.local_entity_encoder(origin_entity, None)
            cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)

        else:
            cand_origin_entity_emb = None
            cand_neighbor_entity_emb = None

        cand_final_emb = self.candidate_encoder(candidate_emb.unsqueeze(0), cand_origin_entity_emb,
                                                cand_neighbor_entity_emb)
        #print("cand_final_emb is{}".format(cand_final_emb.shape)) ### [1,22,400]
        #print("cand_final_emb is{}".format(cand_final_emb))

        # ---------------------------------------------------------------------------------------
        # ----------------------------------------- Score ------------------------------------
        scores = self.click_predictor(cand_final_emb, user_emb).view(-1).cpu().tolist()

        return scores
