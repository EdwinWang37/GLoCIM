import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
import numpy as np
from models.base.function import *

'''
IterableDataset:主要用于大规模数据集，不至于无法一次性完整加载到内存时，可以分批读取数据

torch.nn.functional：torch.nn.functional 是 PyTorch 中的一个模块，
它提供了一系列用于构建神经网络的函数。与 torch.nn 模块中的类不同，torch.nn.functional（通常简称为 F）提供的是函数式接口。
这意味着这些函数直接操作输入数据并返回结果，而不是作为对象的方法被调用。这些函数涵盖了激活函数、损失函数、池化操作等多种神经网络构建块。

torch_geometric:orch_geometric.data 是 PyTorch Geometric (PyG) 库的一部分，这是一个用于图神经网络（GNNs）的 PyTorch 扩展。
在 torch_geometric.data 模块中，主要包含两个关键类：Data 和 Batch。

torch_geometric.utils import subgraph:用于从一个大图中提取子图。
'''




class TrainDataset(IterableDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg):
        super().__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_input = news_input
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = cfg.gpu_num

    def trans_to_nindex(self, nids):#nids是含新闻id的列表，这个函数是获取新闻索引的一个子集
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    #是将一个给定的序列 x 填充到一个固定的长度
    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, np.array(mask, dtype='float32')

    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        clicked_index, clicked_mask = self.pad_to_fix_len(self.trans_to_nindex(click_id), self.cfg.model.his_size)
        clicked_input = self.news_input[clicked_index] #根据索引获取新闻信息

        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        return clicked_input, clicked_mask, candidate_input, label#label有啥用啊？

    def __iter__(self):#IterableDataset的典型用法，适用于数据集太大以至于无法一次性加载到内存
        file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)#map() 是一个内置函数，用于对序列（如列表、元组等）中的每个元素应用一个给定的函数，并返回一个包含结果的迭代器
    
    
class TrainGraphDataset(TrainDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors):
        super().__init__(filename, news_index, news_input, local_rank, cfg)
        self.neighbor_dict = neighbor_dict
        self.news_graph = news_graph.to(local_rank, non_blocking=True)
        #to(local_rank, non_blocking=True)
        #这是 PyTorch 中的一个方法，用于将数据（如张量或模型）移动到不同的设备上，例如从 CPU 移动到 GPU。
        self.batch_size = cfg.batch_size / cfg.gpu_num
        self.entity_neighbors = entity_neighbors




        """weights_dict_path = self.cfg.dataset.train_dir / f"news_weights_dict.bin"
        self.weights_dict = pickle.load(open(weights_dict_path, "rb"))"""

    def line_mapper(self, line, sum_num_news):


        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]#小于等于50条历史点击数据哈，提取最后50个数据
        #print("click_id是{}".format(click_id))
        sess_pos = line[4].split()
        #print("sess_pos是{}".format(sess_pos))
        sess_neg = line[5].split()

        # ------------------ Clicked News ----------------------
        # ------------------ News Subgraph ---------------------

        top_k = len(click_id) #第一跳的数量
        click_idx = self.trans_to_nindex(click_id)
        top_click_idx = list(click_idx)
        source_idx = click_idx

        #引用和复制的区别！对其中一个的改变会影响另一个的变化



        ##one hot neighbors' set

        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)



        """for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_index:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_one_hop_idx)"""

        #long chain selection,first start with one long chain

        #big issues!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!要体现出来吧，这种筛选性质，要不原来太随意了吧，这里设计一个筛选器！！！！
        # one_long_chain = []
        #
        # for news_idx in top_click_idx:
        #     long_chain = [news_idx]
        #     count = 1
        #
        #     # 初始化当前索引为news_idx
        #     current_idx = news_idx
        #
        #     # 循环直到没有更多邻居或达到链的最大长度
        #     while current_idx in self.neighbor_dict and self.neighbor_dict[current_idx]:
        #         # 获取当前节点的邻居列表的第一个邻居
        #         new = self.neighbor_dict[current_idx][0]  # 假设邻居列表至少有一个元素
        #         long_chain.append(new)
        #         count += 1
        #         current_idx = new  # 更新当前索引为新邻居
        #         if count == 10:  # 如果链长度达到10，停止循环
        #             break
        #     one_long_chain.append(long_chain)

        # device = self.news_graph.x.device
        # one_long_chain = torch.tensor(one_long_chain, device = device)
        # print("之前的")
        # print(one_long_chain.size())
        # if one_long_chain.numel() == 0:
        #
        #     print(line)
        #
        # one_long_chain = pad_tensor_to_shape(one_long_chain)
        # print("之后的")
        # print(one_long_chain.size())



        #1.category不同的要mask
        #2.一下特别的临界值，比如说没有10条长链或者说没有50条阅读记录该怎么处理呢


        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, top_k, sum_num_news)

        #mapping是子图中的点击节点到原图的映射，方便后续在不同图的表示中追踪点击节点的身份，在后续的分析和处理中有用
        padded_maping_idx = F.pad(mapping_idx, (self.cfg.model.his_size-len(mapping_idx), 0), "constant", -1) #左边填充若干个-1
        #从左边开始填充

        
        # ------------------ Candidate News ---------------------
        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        # ------------------ Entity Subgraph --------------------
        if self.cfg.model.use_entity:
            origin_entity = candidate_input[:, -3 - self.cfg.model.entity_size:-3]  #[5, 5]
            candidate_neighbor_entity = np.zeros(((self.cfg.npratio+1) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64) # [5*5, 20]
            for cnt,idx in enumerate(origin_entity.flatten()):
                if idx == 0: continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0: continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(self.cfg.npratio+1, self.cfg.model.entity_size *self.cfg.model.entity_neighbors) # [5, 5*10]
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1
            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        return sub_news_graph, padded_maping_idx, candidate_input, candidate_entity, entity_mask, label, \
               sum_num_news+sub_news_graph.num_nodes
        #如果在子图中看到一个节点，可以通过 mapping_idx 知道它在原始图中的哪个位置。

    def build_subgraph(self, subset, k, sum_num_nodes):
        device = self.news_graph.x.device

        if not subset: 
            subset = [0]
            
        subset = torch.tensor(subset, dtype=torch.long, device=device)
        
        unique_subset, unique_mapping = torch.unique(subset, sorted=True, return_inverse=True)#return_inverse,利用这个参数，能索引还原出原来的张量
        subemb = self.news_graph.x[unique_subset]

        sub_edge_index, sub_edge_attr = subgraph(unique_subset, self.news_graph.edge_index, self.news_graph.edge_attr, relabel_nodes=True, num_nodes=self.news_graph.num_nodes)
        #subgraph 的主要功能是根据 subset 中的节点索引，从原始图中提取那些节点以及它们之间的边，从而创建一个新的子图。如果指定了 relabel_nodes=True，则在子图中，节点的索引会被重新标记，使得子图可以作为一个独立的图进行处理。


        sub_news_graph = Data(x=subemb, edge_index=sub_edge_index, edge_attr=sub_edge_attr)
        #这样的设计允许在批处理多个子图时保持节点索引的唯一性和一致性，这对于后续的图神经网络处理是非常重要的。例如，如果我们要将多个这样的子图合并成一个更大的批处理图，那么每个节点的唯一索引就是必需的，以确保正确地关联节点特征和边。

        return sub_news_graph, unique_mapping[:k]+sum_num_nodes #确保每个子图上的节点都具有唯一的索引
    
    def __iter__(self):
        while True:

            clicked_graphs = []
            candidates = []
            mappings = []
            labels = []
            one_long_chains = []

            candidate_entity_list = []
            entity_mask_list = []
            sum_num_news = 0
            with open(self.filename) as f:


                for line in f:
                    # if line.strip().split('\t')[3]:
                    sub_newsgraph, padded_mapping_idx, candidate_input, candidate_entity, entity_mask, label, sum_num_news = self.line_mapper(line, sum_num_news)

                    clicked_graphs.append(sub_newsgraph)
                    candidates.append(torch.from_numpy(candidate_input))
                    mappings.append(padded_mapping_idx)
                    labels.append(label)


                    candidate_entity_list.append(torch.from_numpy(candidate_entity))
                    entity_mask_list.append(torch.from_numpy(entity_mask))

                    #print(torch.cuda.memory_allocated())


                    if len(clicked_graphs) == self.batch_size:


                        batch = Batch.from_data_list(clicked_graphs)
                        #将多个图数据转换成一个批量图数据，将多个图结合成一个大的批处理图
                        candidates = torch.stack(candidates)
                        mappings = torch.stack(mappings)
                        candidate_entity_list = torch.stack(candidate_entity_list)
                        entity_mask_list = torch.stack(entity_mask_list)

                        labels = torch.tensor(labels, dtype=torch.long)
                        yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels
                        #当处理大量数据时，使用 yield 可以节省大量内存。与一次性返回整个列表相比，生成器一次只生成并返回序列中的一个元素
                        clicked_graphs, mappings ,candidates, labels, candidate_entity_list, entity_mask_list,  = [], [], [], [], [], [],
                        sum_num_news = 0

                if (len(clicked_graphs) > 0):#针对最后一次的处理
                    batch = Batch.from_data_list(clicked_graphs) #实际上还是并行的图，不过是放在一起进行批处理，就不需要进行遍历了，但是边的索引需要改变一下
                    candidates = torch.stack(candidates)
                    mappings = torch.stack(mappings)
                    candidate_entity_list = torch.stack(candidate_entity_list)
                    entity_mask_list = torch.stack(entity_mask_list)
                    labels = torch.tensor(labels, dtype=torch.long)

                    yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels
                    f.seek(0) #把文件的读写位置移动到开头


class ValidGraphDataset(TrainGraphDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors, news_entity, news_dict ):
        super().__init__(filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors)
        self.news_graph.x = torch.from_numpy(self.news_input).to(local_rank, non_blocking=True)
        self.news_entity = news_entity
        self.news_dict = news_dict

    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]
        #print("click_id的值为{}".format(click_id))

        # 假设 click_id 是一个列表，你想根据每个ID获取内容
        click_history = []
        for id in click_id:
            if id in self.news_dict:
                click_history.append(self.news_dict[id])
            else:
                click_history.append(None)  # 或者使用其他默认值

        click_idx = self.trans_to_nindex(click_id)
        clicked_entity = self.news_entity[click_idx]
        source_idx = click_idx
        for _ in range(self.cfg.model.k_hops) :
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id), 0)

         # ------------------ Entity --------------------
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
        candidate_index = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        candidate_input = self.news_input[candidate_index]

        if self.cfg.model.use_entity:
            origin_entity = self.news_entity[candidate_index]
            candidate_neighbor_entity = np.zeros((len(candidate_index)*self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
            for cnt,idx in enumerate(origin_entity.flatten()):
                if idx == 0: continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0: continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(len(candidate_index), self.cfg.model.entity_size *self.cfg.model.entity_neighbors)

            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1

            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        batch = Batch.from_data_list([sub_news_graph])

        return batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels, click_history

    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:
                batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels ,click_history= self.line_mapper(line)
            yield batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels, click_history
        #yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels



class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]


    def __len__(self):
        return self.data.shape[0]


