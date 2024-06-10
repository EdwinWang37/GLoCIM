"""
Common utils and tools.
"""
import pickle
import random

import pandas as pd
import torch
import numpy as np
import pyrootutils
from pathlib import Path
import torch.distributed as dist

import importlib
from omegaconf import DictConfig, ListConfig


def seed_everything(seed):
    torch.manual_seed(seed)#一些参数随机性被记录：模型初始权重值，dropout设置种子，样本的随机采样
    torch.cuda.manual_seed_all(seed) #GPU上的随机权重初始化都是相同的，确保不同GPU上的计算都使用相同的随机种子，以确保一致性。
    torch.backends.cudnn.deterministic = True #用于加速卷积的一个深度学习库
    torch.backends.cudnn.benchmark = False#这一行代码关闭了CuDNN的自动优化，以确保运行时不会尝试通过使用不同的算法来优化性能，从而保持可重复性
    random.seed(seed) #Python 标准库的生成器
    np.random.seed(seed) #设置了 NumPy 库中的随机数生成器的种子

def load_model(cfg):
    framework = getattr(importlib.import_module(f"models.{cfg.model.model_name}"), cfg.model.model_name)

    if cfg.model.use_entity:
        entity_dict = pickle.load(open(Path(cfg.dataset.val_dir) / "entity_dict.bin", "rb"))
        entity_emb_path = Path(cfg.dataset.val_dir) / "combined_entity_embedding.vec"
        entity_emb = load_pretrain_emb(entity_emb_path, entity_dict, 100)
    else:
        entity_emb = None

    if cfg.dataset.dataset_lang == 'english':
        word_dict = pickle.load(open(Path(cfg.dataset.train_dir) / "word_dict.bin", "rb"))
        glove_emb = load_pretrain_emb(cfg.path.glove_path, word_dict, cfg.model.word_emb_dim)
    else:
        word_dict = pickle.load(open(Path(cfg.dataset.train_dir) / "word_dict.bin", "rb"))
        glove_emb = len(word_dict)
    model = framework(cfg, glove_emb=glove_emb, entity_emb=entity_emb)

    return model


def save_model(cfg, model, optimizer=None, mark=None):
    file_path = Path(f"{cfg.path.ckp_dir}/{cfg.model.model_name}_{cfg.dataset.dataset_name}_{mark}.pth")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        },
        file_path)
    print(f"Model Saved. Path = {file_path}")


def load_pretrain_emb(embedding_file_path, target_dict, target_dim):
    embedding_matrix = np.zeros(shape=(len(target_dict) + 1, target_dim))
    have_item = []
    if embedding_file_path is not None:
        with open(embedding_file_path, 'rb') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                line = line.split()
                itme = line[0].decode()
                if itme in target_dict:
                    index = target_dict[itme]
                    tp = [float(x) for x in line[1:]]
                    embedding_matrix[index] = np.array(tp)
                    have_item.append(itme)
    print('-----------------------------------------------------')
    print(f'Dict length: {len(target_dict)}')
    print(f'Have words: {len(have_item)}')
    miss_rate = (len(target_dict) - len(have_item)) / len(target_dict) if len(target_dict) != 0 else 0
    print(f'Missing rate: {miss_rate}')
    return embedding_matrix


def reduce_mean(result, nprocs):
    rt = result.detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def pretty_print(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key)+ '\t' + str(value))


def get_root():
    return pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "README.md"],
        pythonpath=True,
        dotenv=True,
    )

def print_model_memory_usage(model):
    print("Model's state_dict:")
    total_params = 0
    for param_tensor in model.state_dict():
        # 获取每个参数的大小（元素总数）
        num_params = model.state_dict()[param_tensor].numel()
        # 每个元素占用的内存（以float32为例，每个元素占4字节）
        param_size = num_params * 4
        total_params += param_size
        print(f"{param_tensor} has {num_params} params: {param_size} bytes")
    print(f"Total memory for model parameters: {total_params} bytes")



class EarlyStopping:
    """
    Early Stopping class
    """

    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = 0.0

    def __call__(self, score):
        """
        The greater score, the better result. Be careful the symbol.
        """
        if score > self.best_score:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_score = score
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better
