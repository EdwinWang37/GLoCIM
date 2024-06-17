from src.models.component.news_encoder import *
import pickle


class filter(nn.Module):
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



    def forward(self,  mapping_idx, entity_mask, label=None, mode = "train"):
        # -------------------------------------- clicked ----------------------------------

        data_dir = {"train": self.cfg.dataset.train_dir, "val": self.cfg.dataset.val_dir, "test": self.cfg.dataset.test_dir}

        news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))
        news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))
        trimmed_news_neighbors_dict = {}

        for key, neighbors_list in news_neighbors_dict.items():
            
            trimmed_neighbors = neighbors_list[:3]

           
            trimmed_news_neighbors_dict[key] = trimmed_neighbors





 
        x_encoded = self.local_news_encoder(x_flatten).view(-1, self.news_dim)


        return

