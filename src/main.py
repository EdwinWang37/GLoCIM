import os.path
from pathlib import Path

import hydra
import math
import wandb
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda import amp
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

from dataload.data_load import load_data
from dataload.data_preprocess import prepare_preprocessed_data, prepare_neighbor_vec_list
from utils.metrics import *
from utils.common import *

### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(model, optimizer, scaler, scheduler, dataloader, local_rank, cfg, early_stopping):
    model.train()
    torch.set_grad_enabled(True)
    mode = "train"
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    first = True
    updated = False


    # print("又开始预处理咯，祝我好运")
    # lsp_state_dict = model.module.local_news_encoder.state_dict()
    # data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
    # torch.save(lsp_state_dict, Path(data_dir["train"]) / "news_local_news_encoder.pth")
    # print("ok了")

    #加载
    with open(Path(data_dir[mode]) / "news_outputs_dict.pt", 'rb') as bf:
        outputs_dict = torch.load(bf, map_location=torch.device('cuda:0'))

    with open(Path(data_dir[mode]) / "trimmed_news_neighbors_dict.bin", 'rb') as file:
        trimmed_news_neighbors_dict = pickle.load(file)

    sum_loss = torch.zeros(1).to(local_rank)
    sum_auc = torch.zeros(1).to(local_rank)



    for cnt, (subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, labels) \
            in enumerate(tqdm(dataloader,
                              total=int(cfg.num_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)), #为了先测试成功，把batch设置为1，原语句是total=int(cfg.num_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)
                              desc=f"[{local_rank}] Training"), start=1):

        #测一下处理1000次的时间和内存
        if cnt == 31:
            print("over了，快看看时间和内存占用情况吧！！！")
            break;
        if cnt > int(cfg.num_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)):
            print("完成{}个epoch的训练了".format(cfg.num_epochs))
            break
        subgraph = subgraph.to(local_rank, non_blocking=True) #将子图部署到特定的GPU上
        mapping_idx = mapping_idx.to(local_rank, non_blocking=True)
        candidate_news = candidate_news.to(local_rank, non_blocking=True)
        labels = labels.to(local_rank, non_blocking=True)
        candidate_entity = candidate_entity.to(local_rank, non_blocking=True)
        entity_mask = entity_mask.to(local_rank, non_blocking=True)

        if(updated == True):
            with open(Path(data_dir[mode]) / "news_outputs_dict.pt", 'rb') as bf:
                outputs_dict = torch.load(bf, map_location=torch.device('cuda:0'))

            file_path = Path(data_dir[mode]) / "trimmed_news_neighbors_dict.bin"
            with open(file_path, 'rb') as file:
                trimmed_news_neighbors_dict = pickle.load(file)
            updated = False

        with amp.autocast():#自动混合精度训练的一部分，可以提高训练速度和效率。它会自动将某些操作从单精度（float32）转换为半精度（float16），这样做的好处是可以加快计算速度，减少内存使用
            bz_loss, y_hat = model(subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask,outputs_dict,trimmed_news_neighbors_dict, labels)


        # Accumulate the gradients，不知道咋加速的，黑盒呗。
        scaler.scale(bz_loss).backward()

        if cnt % cfg.accumulation_steps == 0 or cnt == int(cfg.dataset.pos_count / cfg.batch_size):
            # Update the parameters
            scaler.step(optimizer)
            old_scaler = scaler.get_scale()
            scaler.update()
            new_scaler = scaler.get_scale()
            if new_scaler >= old_scaler:
                scheduler.step()
                ## https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step/164814
            optimizer.zero_grad(set_to_none=True)



        sum_loss += bz_loss.data.float()
        sum_auc += area_under_curve(labels, y_hat)
        if cnt == 10:
            print(torch.cuda.get_device_properties(0))  # 显示第一个GPU的属性
            print(torch.cuda.memory_allocated(0))  # 显示第一个GPU的已分配内存
            print(torch.cuda.memory_cached(0))  # 显示第一个GPU的缓存内存

        # ----------------重新训练筛选邻居节点----------------------
        if cnt ==  3000  or cnt == 6000  or cnt == 9000:
            print("又开始预处理咯，祝我好运")
            lsp_state_dict = model.module.local_news_encoder.state_dict()
            data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
            torch.save(lsp_state_dict, Path(data_dir["train"]) / "news_local_news_encoder.pth")
            prepare_neighbor_vec_list(cfg, 'train')                                                 #忘了with torch.no_grad了吧！！！！！！！！！！！！！！！！！
            updated = True



        # -------------------训练集日志---------------------
        if cnt % cfg.log_steps == 0: #输出训练数据      总共36930次，5个epoch的话
            if local_rank == 0:
                wandb.log({"train_loss": sum_loss.item() / cfg.log_steps, "train_auc": sum_auc.item() / cfg.log_steps})
            print('[{}] Ed: {}, average_loss: {:.5f}, average_acc: {:.5f}'.format(
                 local_rank, cnt * cfg.batch_size, sum_loss.item() / cfg.log_steps, sum_auc.item() / cfg.log_steps))
            #if best_auc < sum_auc.item()/ cfg.log_steps:
                    #beyond = True
                    #best_auc = sum_auc.item() / cfg.log_steps
                    #best_loss = sum_loss.item() / cfg.log_steps
            if math.isnan(sum_loss.item() / cfg.log_steps):
                print("拉倒了，一首凉凉送给自己！")
                break

            sum_loss.zero_()
            sum_auc.zero_()

        # #保存训练集的东东
        # x = cfg.dataset.pos_count // cfg.batch_size + 1
        # if cnt > x and beyond:
        #     print("------------------------------------------------")
        #     print(f"Better Result!")
        #     print("best_auc为：{}，best_loss为：{}".format(best_auc,best_loss))
        #     print("------------------------------------------------")
        #     if local_rank == 0:
        #         save_model(cfg, model, optimizer, f"{cfg.ml_label}_auc{best_auc}")
        #     beyond = False

        #if cnt > int(cfg.val_skip_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)) and cnt % cfg.val_steps == 0:

        #测试集启动启动启动！！！
        if cnt % cfg.val_steps == 0 and cnt > 8000:
            if first == True:
                prepare_neighbor_vec_list(cfg, 'val')
                first = False
            res = val(model, local_rank, cfg)
            model.train()
            print("--------------------睁大眼睛看好了！-----------------------")
            if local_rank == 0:
                pretty_print(res)
                wandb.log(res)
            print("--------------------是骡子还是马啊？？？--------------------")
            early_stop, get_better = early_stopping(res['auc'])

            if early_stop:
                print("Early Stop.")
                break
            elif get_better:
                print(f"Better Result!")
                if local_rank == 0:
                    save_model(cfg, model, optimizer, f"{cfg.ml_label}_auc{res['auc']}")
                    wandb.run.summary.update({"best_auc": res["auc"], "best_mrr": res['mrr'],
                                              "best_ndcg5": res['ndcg5'], "best_ndcg10": res['ndcg10']})



def val(model, local_rank, cfg):

    model.eval()

    mode = "val"
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
    # 加载多余的这俩数据
    with open(Path(data_dir[mode]) / "news_outputs_dict.pt", 'rb') as bf:
        outputs_dict = torch.load(bf, map_location=torch.device('cuda:0'))
        #print("outputs_dict的维度是{}".format(outputs_dict.shape))

    file_path = Path(data_dir[mode]) /"trimmed_news_neighbors_dict.bin"
    with open(file_path, 'rb') as file:
        trimmed_news_neighbors_dict = pickle.load(file)


    dataloader = load_data(cfg, mode='val', model=model, local_rank=local_rank)


    #sample_ratio = 0.005
    tasks = []
    with torch.no_grad():
        for cnt, (subgraph, mappings, clicked_entity, candidate_input, candidate_entity, entity_mask, labels, click_history) \
                in enumerate(tqdm(dataloader,
                                  total=int(cfg.dataset.val_len / cfg.gpu_num),
                                  desc=f"[{local_rank}] Validating")):
            # cnt += 1
            candidate_emb = torch.FloatTensor(np.array(candidate_input)).to(local_rank, non_blocking=True)
            candidate_entity = candidate_entity.to(local_rank, non_blocking=True)
            entity_mask = entity_mask.to(local_rank, non_blocking=True)
            clicked_entity = clicked_entity.to(local_rank, non_blocking=True)

            scores = model.module.validation_process(subgraph, mappings, clicked_entity, candidate_emb,
                                                     candidate_entity, entity_mask, outputs_dict, trimmed_news_neighbors_dict, click_history)

            tasks.append((labels.tolist(), scores))

    #开启线程池，把计算任务分发
    with mp.Pool(processes=cfg.num_workers) as pool:
        results = pool.map(cal_metric, tasks)
    val_auc, val_mrr, val_ndcg5, val_ndcg10 = np.array(results).T

    # barrier
    torch.distributed.barrier()#同步所有的计算节点

    #平均值
    reduced_auc = reduce_mean(torch.tensor(np.nanmean(val_auc)).float().to(local_rank), cfg.gpu_num)
    reduced_mrr = reduce_mean(torch.tensor(np.nanmean(val_mrr)).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg5 = reduce_mean(torch.tensor(np.nanmean(val_ndcg5)).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg10 = reduce_mean(torch.tensor(np.nanmean(val_ndcg10)).float().to(local_rank), cfg.gpu_num)

    res = {
        "auc": reduced_auc.item(),
        "mrr": reduced_mrr.item(),
        "ndcg5": reduced_ndcg5.item(),
        "ndcg10": reduced_ndcg10.item(),
    }

    return res

#有助于多进程任务，使用多个gpu同时训练时会遇到
def main_worker(local_rank, cfg):
    # -----------------------------------------Environment Initial
    seed_everything(cfg.seed)
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=cfg.gpu_num,
                            rank=local_rank)

    # -----------------------------------------Dataset & Model Load
    num_training_steps = int(cfg.num_epochs * cfg.dataset.pos_count / (cfg.batch_size * cfg.accumulation_steps))#accumulation的作用：积累梯度，如果=2，那么说明每两个batch更新一次梯度
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio + 1)#3x236344/32x1 = 22157     #2215
    train_dataloader = load_data(cfg, mode='train', local_rank=local_rank)
    #dataloader加载完毕
    model: object = load_model(cfg).to(local_rank) #模型上显卡
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    lr_lambda = lambda step: 0.2 if step > num_warmup_steps else step / num_warmup_steps * 0.2 #学习率进行线性增加 ???
    scheduler = LambdaLR(optimizer, lr_lambda)#调度器，根据上述的学习率增加逻辑



    # ------------------------------------------Load Checkpoint & optimizer
    if cfg.load_checkpoint:
        file_path = Path(f"{cfg.path.ckp_dir}/{cfg.model.model_name}_{cfg.dataset.dataset_name}_default_auc0.6790186762809753.pth")
        print(file_path)
        print("--------------------------")
        checkpoint = torch.load(file_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])  # After Distributed strict取消是因为消融实验呢
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) #分布式

    ######先保存第一个好吧

    # lsp_state_dict = model.module.local_news_encoder.state_dict()
    # data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
    # torch.save(lsp_state_dict, Path(data_dir["train"]) / "news_local_news_encoder.pth")
    # print("baocunchenggong")
    # return 0

    ##########
    optimizer.zero_grad(set_to_none=True) #梯度清0
    scaler = amp.GradScaler()  #利用混和精度减少GPU计算的手段

    # ------------------------------------------Main Start
    early_stopping = EarlyStopping(cfg.early_stop_patience) ##模型的性能在5个连续的周期内都没改进，则训练就会停止

    #用于跟踪机器学习实验，记录指标，输出，模型权重等
    if local_rank == 0:
        wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                   project=cfg.logger.exp_name, name=cfg.logger.run_name)
        print(model)

    num = 0
    # for _ in tqdm(range(1, cfg.num_epochs + 1), desc="Epoch"):
    train(model, optimizer, scaler, scheduler, train_dataloader, local_rank, cfg, early_stopping)
    #scaler用于梯度缩放
    #scheduler用于学习率调节


    if local_rank == 0:
        wandb.finish()


@hydra.main(version_base="1.2", config_path=os.path.join(get_root(), "configs"), config_name="small")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)#固定随机种子
    cfg.gpu_num = torch.cuda.device_count()
    prepare_preprocessed_data(cfg)
    print("开始训练！")
    mp.spawn(main_worker, nprocs=cfg.gpu_num, args=(cfg,)) #mp.spawn主要用于多进程处理


if __name__ == "__main__":
    main()

