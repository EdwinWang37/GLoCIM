import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GatedGraphConv

from models.component.candidate_encoder import *
from models.component.click_encoder import ClickEncoder
from models.component.entity_encoder import EntityEncoder, GlobalEntityEncoder
from models.component.nce_loss import NCELoss
from models.component.news_encoder import *
from models.component.user_encoder import *

from models.component.global_news_long_encoder import GlobalNewsLongEncoder

from models.component.feature_gate import FeatureGate


class LSP-GNR(nn.Module):
    def __init__(self, cfg, glove_emb=None, entity_emb=None):
        super().__init__()

        self.cfg = cfg
        self.use_entity = cfg.model.use_entity

        attention_input_dim = self.cfg.model.head_num * self.cfg.model.head_dim
        self.news_dim = cfg.model.head_num * cfg.model.head_dim  # 20 * 20 = 400
        self.entity_dim = cfg.model.entity_emb_dim  # 100

        # -------------------------- Model --------------------------
        # News Encoder
        self.local_news_encoder = NewsEncoder(cfg, glove_emb)

        # GCN
        self.global_news_short_encoder = Sequential('x, index', [
            (GatedGraphConv(self.news_dim, num_layers=3, aggr='add'), 'x, index -> x'),
        ])

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

    def forward(self, subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, outputs_dict,
                trimmed_news_neighbors_dict, label=None):

        # -------------------------------------- clicked ----------------------------------
        # mapping_idx是为了解决子图中的不重复节点和点击中的重复节点的问题，通过unique_mapping的对照，即可搭建subgarph和click_index的映射
        mask = mapping_idx != -1  # mapping_idx 的维度是 【batch(user),clicked_news】,so 这个维度问题可以效仿
        mapping_idx[mapping_idx == -1] = 0
        # print("mapping_idx的维度是{}".format(mapping_idx.shape))

        batch_size, num_clicked, token_dim = mapping_idx.shape[0], mapping_idx.shape[1], candidate_news.shape[-1]
        clicked_entity = subgraph.x[mapping_idx, -8:-3]  # [16, 50, 5]
        click_history = subgraph.x[mapping_idx, -1]  # 不对[16, 50]
        # print("------------mapping_idx是多少{}".format(mapping_idx))
        # print("------------click_history是多少{}".format(click_history)) #click_his的维度是【16,50】      有瑕疵，其中1469代表没有
        # print("------------click_history的维度多少{}".format(click_history.shape))
        # print("------------clicked_entity的维度多少{}".format(clicked_entity.shape))
        # print("------------subgraph.x[0]是{}".format(subgraph.x[0]))
        # 奶奶滴！终于知道了！为啥这里会变成一个固定的1469

        # News
        # print("subgraph.x维度是:{}".format(subgraph.x.shape))
        x_flatten = subgraph.x.view(1, -1, token_dim)
        x_encoded = self.local_news_encoder(x_flatten).view(-1,
                                                            self.news_dim)  # [batch, news, news_dim] 32, 50, 400 -> 32 * 50, 400

        # print("x_encoded是：")
        # print(x_encoded)
        # print("x_encoded维度是：")
        # print(x_encoded.shape)

        # 短链
        global_short_emb = self.global_news_short_encoder(x_encoded, subgraph.edge_index)
        # print("global_short_emb的维度是{}".format(global_short_emb.shape))
        # print("第一阶段测试成功！")

        ######有点担心category是否真的嵌入了，以及筛选的时候是否真的用上饿了，以及如果长链不到位的话，那么哪些异常值怎么处理，以及怎么把长链中途的其他链去掉
        # 挑选出新闻并填充咯
        clicked_origin_emb = x_encoded[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked,
                                                                                                self.news_dim)
        # print("clicked_origin_emb的维度是{}".format(clicked_origin_emb.shape))
        # print("clicked_origin_emb[0]是{}".format(clicked_origin_emb[0]))

        # 挑选出新闻并填充咯
        clicked_short_graph_emb = global_short_emb[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size,
                                                                                                            num_clicked,
                                                                                                            self.news_dim)
        #print("————————————clicked_short_graph_emb的维度是{}-----------------------".format(
            clicked_short_graph_emb.shape))
        # print("clicked_origin_emb[0]的值为{}".format(clicked_origin_emb[0]))

        # print("------------1------------")

        # print("clicked_origin_emb的维度为{}".format(clicked_origin_emb.shape))
        # print("clicked_origin_emb[0]的值为：{}".format(clicked_origin_emb[0]))

        # print("click_history的维度为{}".format(click_history.shape))
        # print("click_history[0]的值为：{}".format(click_history[0]))

        # print("outputs_dict的维度为{}".format(outputs_dict.shape))
        # print("outputs_dict[1]的值为：{}".format(outputs_dict[1]))
        # print("outputs_dict[0]的值为：{}".format(outputs_dict[0]))  #0不好用？？？

        # 长链

        global_long_emb = self.global_news_long_encoder(clicked_origin_emb, click_history, outputs_dict,
                                                        trimmed_news_neighbors_dict)  # [16,50,400] 16记录，50条阅读记录，
        # print("第二阶段测试成功！global_long_emb完成了完成了,皆大欢喜，global_long_emb的维度是{}".format(global_long_emb.shape))

        # 假设clicked_short_graph_emb和global_long_emb是已经定义好的张量
        # clicked_short_graph_emb, global_long_emb: [16, 50, 400]
        click_fuse_vec = self.feature_gate(clicked_short_graph_emb, global_long_emb)  # 输出也将是[16, 50, 400]
        # print("第三阶段测试成功！click_fuse_vec完成了完成了,更开心了哦耶，click_fuse_vec的维度是{}".format(click_fuse_vec.shape))
        # print("------------------------------2---------------------------------------")
        # clicked_long_graph_emb = global_long_emb[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size,
        #                                                                                              num_clicked,
        #                                                                                              self.news_dim)
        # Attention pooling
        if self.use_entity:
            clicked_entity = self.local_entity_encoder(clicked_entity, None)
            print()
        else:
            clicked_entity = None

        clicked_total_emb = self.click_encoder(clicked_origin_emb, click_fuse_vec, clicked_entity)  # 局部新闻+全局+局部实体
        user_emb = self.user_encoder(clicked_total_emb, mask)
        # print("1")
        # ----------------------------------------- Candidate------------------------------------
        cand_title_emb = self.local_news_encoder(candidate_news)  # [8, 5, 400]
        # print("2")
        if self.use_entity:
            origin_entity, neighbor_entity = candidate_entity.split(
                [self.cfg.model.entity_size, self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)

            cand_origin_entity_emb = self.local_entity_encoder(origin_entity, None)
            cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)
            print("3")

            # cand_entity_emb = self.entity_encoder(candidate_entity, entity_mask).view(batch_size, -1, self.news_dim) # [8, 5, 400]
        else:
            cand_origin_entity_emb, cand_neighbor_entity_emb = None, None

        # print("cand_title_emb的维度是：{}".format(cand_title_emb.shape))
        # print("cand_origin_entity_emb的维度是：{}".format(cand_origin_entity_emb.shape))
        # print("cand_neighbor_entity_emb的维度是：{}".format(cand_neighbor_entity_emb.shape))
        # print("---------------------------------------------------------------------")

        cand_final_emb = self.candidate_encoder(cand_title_emb, cand_origin_entity_emb, cand_neighbor_entity_emb)
        # ----------------------------------------- Score ------------------------------------
        score = self.click_predictor(cand_final_emb, user_emb)
        loss = self.loss_fn(score, label)
        # print("真能运行出来？")
        # print("score的值为：{}".format(score))
        # print("loss的值为：{}".format(loss))
        return loss, score

    def validation_process(self, subgraph, mappings, clicked_entity, candidate_emb, candidate_entity, entity_mask,
                           outputs_dict, trimmed_news_neighbors_dict):

        batch_size, num_news, news_dim = 1, len(mappings), candidate_emb.shape[-1]

        title_graph_emb = self.global_news_encoder(subgraph.x, subgraph.edge_index)
        clicked_graph_emb = title_graph_emb[mappings, :].view(batch_size, num_news, news_dim)
        clicked_origin_emb = subgraph.x[mappings, :].view(batch_size, num_news, news_dim)

        # -------------------------------------Attention Pooling--------------------------------
        if self.use_entity:
            clicked_entity_emb = self.local_entity_encoder(clicked_entity.unsqueeze(0), None)
        else:
            clicked_entity_emb = None

        clicked_final_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity_emb)

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
        # ---------------------------------------------------------------------------------------
        # ----------------------------------------- Score ------------------------------------
        scores = self.click_predictor(cand_final_emb, user_emb).view(-1).cpu().tolist()

        return scores
