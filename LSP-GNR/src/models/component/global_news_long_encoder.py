import copy

import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential, GCNConv
from pathlib import Path
import torch.nn.functional as F
import copy

import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
import pickle



class GlobalNewsLongEncoder(nn.Module):
    def __init__(self, cfg, glove_emb=None):
        super().__init__()
        token_emb_dim = cfg.model.word_emb_dim
        self.news_dim = cfg.model.head_num * cfg.model.head_dim



        self.attt = Sequential('x, mask', [ #【16，50，400】 ->[16, 400]
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            (MultiHeadAttention(self.news_dim ,
                                self.news_dim ,
                                self.news_dim ,
                                cfg.model.head_num,
                                cfg.model.head_dim), 'x,x,x,mask -> x'),  # 20 * 20
            ReshapeLayer((-1, 50, 10, self.news_dim)),
            ReshapeLayer((-1, 10, self.news_dim)),
            nn.LayerNorm(self.news_dim),  # 400
            nn.Dropout(p=cfg.dropout_probability),  # 0.2

            (AttentionPooling(self.news_dim,
                              cfg.model.attention_hidden_dim), 'x,mask -> x'),
            nn.LayerNorm(self.news_dim),  # 400

        ])

    # 一个索引对应的映射news_input,然后才能注意力是吧,但是有一个文件了其实，那么实际上就是把那个文件加载进来，然后相乘不就行了吗？还是说不行呢？还是作为一个整体呢？okk，那么我选择最简化方法了！先采取最简单的方法，后面不行再加复杂度
    # 先选10个邻居，然后trans合并（注意填充的0向量不参与筛选）
    # mapping_idx不行的哦，需要的是点击新闻的索引哈。
    #######################################neighbor_dict索引差了1！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    def forward(self, news_input, click_history, outputs_dict, trimmed_news_neighbors_dict,mask = None):

        # news_input: [16, 50, 400]
        # click_history: [16, 50]
        # outputs_dict: [51282, 3, 400]

        #print("trimmed_news_neighbors_dict[3391]是：{}".format(trimmed_news_neighbors_dict[3391]))
        #print("trimmed_news_neighbors_dict[1][2]是：{}".format(trimmed_news_neighbors_dict[1][2]))
        batch_size, news_num, feature_size = news_input.shape
        results = torch.zeros(batch_size, news_num, 10, feature_size, device=news_input.device) #16, 50, 10, 400
        for batch_idx in range(batch_size):
            for news_idx in range(news_num):
                if torch.any(news_input[batch_idx, news_idx] != 0):
                    current_vector = news_input[batch_idx, news_idx].unsqueeze(0)  # [1, 400]
                    indices = click_history[batch_idx, news_idx].item()
                    use_zero_vector = False  # 新增标志，初始化为False
                    for selection_idx in range(6):
                        if not use_zero_vector:        #查一下indices-1和outputs中的对应关系哈！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                            vectors_group = outputs_dict[indices - 1]  # [3, 400]
                            scores = torch.matmul(vectors_group, current_vector.t())  # [3, 1]
                            max_val, max_idx = torch.max(scores, 0)

                            if max_val == 0:
                                non_zero_indices = torch.nonzero(scores.squeeze(), as_tuple=True)[0]
                                if len(non_zero_indices) > 0:
                                    non_zero_scores = scores[non_zero_indices]
                                    max_val_non_zero, max_idx_non_zero = torch.max(non_zero_scores, 0)
                                    max_idx = non_zero_indices[max_idx_non_zero]
                                    if indices in trimmed_news_neighbors_dict:
                                        # 使用 item() 方法从单元素张量中提取值
                                        neighbor_indices = trimmed_news_neighbors_dict[indices]
                                        indices = neighbor_indices[max_idx.item()]  # 假设 max_idx 也是一个张量
                                    else:
                                        print(f"Key {indices} not found in trimmed_news_neighbors_dict")
                                else:
                                    use_zero_vector = True  # 找不到非零值，激活零向量使用标志
                                    #print("坏了啊！！！撒日朗！！！")

                        # 如果激活了使用零向量标志或本来就应该使用零向量
                        if use_zero_vector:
                            selected_vector = torch.zeros_like(current_vector.squeeze())
                        else:
                            selected_vector = vectors_group[max_idx.item()]

                        results[batch_idx, news_idx, selection_idx] = selected_vector
        #print("results.shape的值为{}".format(results.shape))#results的维度是[16, 50, 10 400]
                        # 更新click_history为选定向量的新索引（如果需要）
                        # click_history[batch_idx, news_idx] = new_index

        if news_num < 50:
            # 需要填充到50
            padding_size = 50 - news_num
            padded_results = F.pad(results, (0, 0, 0, 0, 0, padding_size, 0, 0), "constant", 0)
        else:
            padded_results = results
        results_ = padded_results.view(batch_size, 50 * 10, self.news_dim)  # [16, 500, 400]
        #print("--------------------------------------------------")
        #print("results_的维度以前是{}".format(results.shape))
        #print("results_的维度现在是{}".format(padded_results.shape))

        final_output = self.attt(results_, mask)
        #print("final_output是{}".format(final_output.shape)) #[50, 400]
        #print("final_output是{}".format(final_output))
        # 应用多头注意力机制
        # 这里简化处理，没有使用mask，实际情况可能需要考虑mask


        return final_output.view(batch_size, 50, self.news_dim)


