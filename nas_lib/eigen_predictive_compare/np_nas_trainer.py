import torch
from gnn_lib.data import Data, Batch
import random
from .utils import gen_batch_idx
from nas_lib.utils.metric_logger import MetricLogger
import torch.nn.functional as F
from nas_lib.predictors_compare.NP_NAS.gcn_np import NasBenchGCNNnpAgent
from nas_lib.predictors_compare.NP_NAS.gcn_np_cas import NasBenchGCNNnpCasAgent
from nas_lib.utils.utils_data import nasbench2graph_reverse
import numpy as np
import time
from nas_lib.utils.comm import get_spearmanr_coorlection, get_kendalltau_coorlection


def make_agent_optimizer(model, base_lr):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_lr
        weight_decay = 0.001
        if "bias" in key:
            lr = base_lr
            weight_decay = 0.0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.Adam(params, base_lr, (0.0, 0.9))
    return optimizer


def lr_step(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class NasBenchGcnCascadeNnpTrainer:
    def __init__(self, device=None, epochs=300, input_dim=6, logger=None):
        self.stage1 = NasBenchGCNNnpCasAgent(input_dim=input_dim)
        self.stage2 = NasBenchGCNNnpAgent(input_dim=input_dim)
        self.criterion = torch.nn.MSELoss()
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.optimizer_cls = make_agent_optimizer(self.stage1, base_lr=0.0002)
        self.optimizer_regress = make_agent_optimizer(self.stage2, base_lr=0.0001)
        self.scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_cls,
                                                                        eta_min=0, T_max=20)
        self.scheduler_regress = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_regress,
                                                                            eta_min=0, T_max=20)
        self.device = device
        self.epochs = epochs
        self.stage1.to(device)
        self.stage2.to(device)
        self.logger = logger

    def fit(self, edge_index, node_feature, edge_index_reverse, node_feature_reverse, val_accuracy, val_accuracy_cls):
        meters_cls = MetricLogger(delimiter="  ")
        meters_regerss = MetricLogger(delimiter="  ")
        self.stage1.train()
        self.stage2.train()
        for epoch in range(self.epochs):
            idx_list = list(range(len(edge_index)))
            random.shuffle(idx_list)
            batch_idx_list = gen_batch_idx(idx_list, 10)

            for i, batch_idx in enumerate(batch_idx_list):
                data_list = []
                data_list_reverse = []
                target_list_cls = []
                for idx in batch_idx:
                    g_d = Data(edge_index=edge_index[idx].long(), x=node_feature[idx].float())
                    data_list.append(g_d)
                    g_d_reverse = Data(edge_index=edge_index_reverse[idx].long(), x=node_feature_reverse[idx].float())
                    data_list_reverse.append(g_d_reverse)
                    target_list_cls.append(val_accuracy_cls[idx])

                val_cls_tensor = torch.tensor(target_list_cls, dtype=torch.long)
                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.device)
                val_cls_tensor = val_cls_tensor.to(self.device)
                batch_nodes, batch_edge_idx, batch_idx_g = batch.x, batch.edge_index, batch.batch
                batch_nodes = F.normalize(batch_nodes, p=2, dim=-1)
                batch_reverse = Batch.from_data_list(data_list_reverse)
                batch_reverse = batch_reverse.to(self.device)
                batch_nodes_reverse, batch_edge_idx_reverse, batch_idx_reverse = batch_reverse.x, \
                                                                                 batch_reverse.edge_index, \
                                                                                 batch_reverse.batch
                batch_nodes_reverse = F.normalize(batch_nodes_reverse, p=2, dim=-1)

                pred_cls = self.stage1(batch_nodes, batch_edge_idx, batch_idx_g, batch_nodes_reverse,
                                       batch_edge_idx_reverse).squeeze()
                loss_stage1 = self.criterion_ce(pred_cls, val_cls_tensor)
                self.optimizer_cls.zero_grad()
                loss_stage1.backward()
                self.optimizer_cls.step()
                self.scheduler_cls.step(epoch + int(i/30))
                meters_cls.update(loss=loss_stage1.item())

                if pred_cls.dim() == 1:
                    pred_cls.unsequeeze_(dim=0)

                pred_max = torch.argmax(pred_cls, dim=1)
                if torch.sum(pred_max) == 0:
                    continue
                data_list = []
                data_list_reverse = []
                target_list = []
                for k, idx in enumerate(batch_idx):
                    if pred_max[k] == 0:
                        continue
                    g_d = Data(edge_index=edge_index[idx].long(), x=node_feature[idx].float())
                    data_list.append(g_d)
                    g_d_reverse = Data(edge_index=edge_index_reverse[idx].long(), x=node_feature_reverse[idx].float())
                    data_list_reverse.append(g_d_reverse)
                    target_list.append(val_accuracy[idx])
                val_tensor = torch.tensor(target_list, dtype=torch.float32)
                val_tensor = val_tensor.to(self.device)
                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.device)
                batch_nodes, batch_edge_idx, batch_idx_G = batch.x, batch.edge_index, batch.batch
                batch_nodes = F.normalize(batch_nodes, p=2, dim=-1)
                batch_reverse = Batch.from_data_list(data_list_reverse)
                batch_reverse = batch_reverse.to(self.device)
                batch_nodes_reverse, batch_edge_idx_reverse, batch_idx_reverse = batch_reverse.x, \
                                                                                 batch_reverse.edge_index, \
                                                                                 batch_reverse.batch
                batch_nodes_reverse = F.normalize(batch_nodes_reverse, p=2, dim=-1)
                pred_regress = self.stage2(batch_nodes, batch_edge_idx, batch_idx_G, batch_nodes_reverse,
                                           batch_edge_idx_reverse).squeeze()
                if pred_regress.dim() == 0:
                    loss_stage2 = self.criterion(pred_regress, val_tensor[0])
                else:
                    loss_stage2 = self.criterion(pred_regress, val_tensor)
                self.optimizer_regress.zero_grad()
                loss_stage2.backward()
                self.optimizer_regress.step()
                self.scheduler_regress.step(epoch + int(i/30))
                meters_regerss.update(loss=loss_stage2.item())
        return meters_cls.meters['loss'].avg, meters_regerss.meters['loss'].avg

    def pred(self,  edge_index, node_feature, edge_index_reverse, node_feature_reverse):
        pred_list = []
        idx_list = list(range(len(edge_index)))
        self.stage1.eval()
        self.stage2.eval()
        batch_idx_list = gen_batch_idx(idx_list, 32)
        with torch.no_grad():
            for i, batch_idx in enumerate(batch_idx_list):
                data_list = []
                data_list_reverse = []
                for idx in batch_idx:
                    g_d = Data(edge_index=edge_index[idx].long(), x=node_feature[idx].float())
                    data_list.append(g_d)
                    g_d_reverse = Data(edge_index=edge_index_reverse[idx].long(),
                                       x=node_feature_reverse[idx].float())
                    data_list_reverse.append(g_d_reverse)
                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.device)
                batch_nodes, batch_edge_idx, batch_idx_g = batch.x, batch.edge_index, batch.batch
                batch_nodes = F.normalize(batch_nodes, p=2, dim=-1)
                batch_reverse = Batch.from_data_list(data_list_reverse)
                batch_reverse = batch_reverse.to(self.device)
                batch_nodes_reverse, batch_edge_idx_reverse, batch_idx_reverse = batch_reverse.x, \
                                                                                 batch_reverse.edge_index, \
                                                                                 batch_reverse.batch
                batch_nodes_reverse = F.normalize(batch_nodes_reverse, p=2, dim=-1)
                pred_cls = self.stage1(batch_nodes, batch_edge_idx, batch_idx_g, batch_nodes_reverse,
                                       batch_edge_idx_reverse).squeeze()
                if pred_cls.dim() == 1:
                    pred_cls.unsqueeze_(dim=0)
                pred_max = torch.argmax(pred_cls, dim=1)
                if pred_max.dim() == 0:
                    pred_max.unsqueeze_(dim=0)
                if torch.sum(pred_max) == 0:
                    continue
                data_list = []
                data_list_reverse = []
                for k, idx in enumerate(batch_idx):
                    if pred_max[k] == 0:
                        continue
                    g_d = Data(edge_index=edge_index[idx].long(), x=node_feature[idx].float())
                    data_list.append(g_d)
                    g_d_reverse = Data(edge_index=edge_index_reverse[idx].long(), x=node_feature_reverse[idx].float())
                    data_list_reverse.append(g_d_reverse)

                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.device)
                batch_nodes, batch_edge_idx, batch_idx_g = batch.x, batch.edge_index, batch.batch
                batch_nodes = F.normalize(batch_nodes, p=2, dim=-1)
                batch_reverse = Batch.from_data_list(data_list_reverse)
                batch_reverse = batch_reverse.to(self.device)
                batch_nodes_reverse, batch_edge_idx_reverse, batch_idx_reverse = batch_reverse.x, \
                                                                                 batch_reverse.edge_index, \
                                                                                 batch_reverse.batch
                batch_nodes_reverse = F.normalize(batch_nodes_reverse, p=2, dim=-1)
                pred_regress = self.stage2(batch_nodes, batch_edge_idx, batch_idx_g, batch_nodes_reverse,
                                           batch_edge_idx_reverse).squeeze()
                pred = torch.zeros_like(pred_max, dtype=torch.float32)
                if pred_regress.dim() == 0:
                    pred_regress.unsqueeze_(dim=0)
                # print(pred_max.size(), pred_regress.size(), pred.size())
                counter = 0
                for j in range(pred.size(0)):
                    if pred_max[j] == 0:
                        pred[j] = 0
                    else:
                        pred[j] = pred_regress[counter]
                        counter += 1
                if len(pred.size()) == 0:
                    pred.unsqueeze_(0)
                pred_list.append(pred)
        return torch.cat(pred_list, dim=0)

    # def fit_g(self, ytrain, train_data, ytest, test_data):
    #     arch_data_edge_idx_list = []
    #     arch_data_node_f_list = []
    #     arch_data_edge_idx_reverse_list = []
    #     arch_data_node_f_reverse_list = []
    #
    #     for arch in train_data:
    #         edge_index, node_f = nasbench2graph_reverse(arch)
    #         arch_data_edge_idx_list.append(edge_index)
    #         arch_data_node_f_list.append(node_f)
    #
    #     for arch in train_data:
    #         edge_index, node_f = nasbench2graph_reverse(arch, reverse=True)
    #         arch_data_edge_idx_reverse_list.append(edge_index)
    #         arch_data_node_f_reverse_list.append(node_f)
    #
    #     val_accuracy = np.array([100 - d for d in ytrain])
    #     val_accuracy_cls = []
    #     for d in val_accuracy:
    #         if d >= 91:
    #             val_accuracy_cls.append(1)
    #         else:
    #             val_accuracy_cls.append(0)
    #     val_accuracy_cls = np.array(val_accuracy_cls)
    #
    #     candiate_edge_list = []
    #     candiate_node_list = []
    #     candiate_arch_data_edge_idx_reverse_list = []
    #     candiate_arch_data_node_f_reverse_list = []
    #     for cand in test_data:
    #         edge_index, node_f = nasbench2graph_reverse(cand)
    #         candiate_edge_list.append(edge_index)
    #         candiate_node_list.append(node_f)
    #
    #         edge_index_reverse, node_f_reverse = nasbench2graph_reverse(cand, reverse=True)
    #         candiate_arch_data_edge_idx_reverse_list.append(edge_index_reverse)
    #         candiate_arch_data_node_f_reverse_list.append(node_f_reverse)
    #
    #     val_test_accuracy = np.array([100 - d for d in ytest])
    #     loss_val_train = self.fit(arch_data_edge_idx_list, arch_data_node_f_list, arch_data_edge_idx_reverse_list,
    #                               arch_data_node_f_reverse_list, val_accuracy, val_accuracy_cls)
    #     predictions = self.pred(arch_data_edge_idx_list,
    #                             arch_data_node_f_list,
    #                             arch_data_edge_idx_reverse_list,
    #                             arch_data_node_f_reverse_list,
    #                             )
    #     predictions = predictions.cpu().numpy()


class NasBenchGcnNnpTrainer:
    def __init__(self, device=None, epochs=300, input_dim=6, params=None, logger=None):
        self.stage1 = NasBenchGCNNnpAgent(input_dim=input_dim)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = make_agent_optimizer(self.stage1, base_lr=params['lr'])
        self.lr = params['lr']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=0, T_max=20)
        self.steps = 0
        self.epochs = epochs
        self.device = torch.device(f'cuda:{device}')
        self.stage1.to(device)
        self.logger = logger

    def fit(self, edge_index, node_feature, edge_index_reverse, node_feature_reverse, val_accuracy):
        meters = MetricLogger(delimiter="  ")
        self.stage1.train()
        for epoch in range(self.epochs):
            idx_list = list(range(len(edge_index)))
            random.shuffle(idx_list)
            batch_idx_list = gen_batch_idx(idx_list, 10)
            for i, batch_idx in enumerate(batch_idx_list):
                data_list = []
                data_list_reverse = []
                target_list = []
                for idx in batch_idx:
                    g_d = Data(edge_index=edge_index[idx].long(), x=node_feature[idx].float())
                    data_list.append(g_d)
                    g_d_reverse = Data(edge_index=edge_index_reverse[idx].long(), x=node_feature_reverse[idx].float())
                    data_list_reverse.append(g_d_reverse)
                    target_list.append(val_accuracy[idx])

                val_tensor = torch.tensor(target_list, dtype=torch.float32)
                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.device)
                val_tensor = val_tensor.to(self.device)
                batch_nodes, batch_edge_idx, batch_idx = batch.x, batch.edge_index, batch.batch
                batch_nodes = F.normalize(batch_nodes, p=2, dim=-1)

                batch_reverse = Batch.from_data_list(data_list_reverse)
                batch_reverse = batch_reverse.to(self.device)
                batch_nodes_reverse, batch_edge_idx_reverse, batch_idx_reverse = batch_reverse.x, \
                                                                                 batch_reverse.edge_index, \
                                                                                 batch_reverse.batch
                batch_nodes_reverse = F.normalize(batch_nodes_reverse, p=2, dim=-1)

                pred = self.stage1(batch_nodes, batch_edge_idx, batch_idx, batch_nodes_reverse,
                                   batch_edge_idx_reverse)
                pred = pred.squeeze()
                loss = self.criterion(pred, val_tensor)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(epoch + int(i/30))
                meters.update(loss=loss.item())
        return meters.meters['loss'].avg

    def pred(self, g_data_list, g_data_list_reverse):
        pred_list = []
        idx_list = list(range(len(g_data_list)))
        self.stage1.eval()
        batch_idx_list = gen_batch_idx(idx_list, 32)
        with torch.no_grad():
            for batch_idx in batch_idx_list:
                data_list = [g_data_list[idx] for idx in batch_idx]
                data_list_reverse = [g_data_list_reverse[idx] for idx in batch_idx]

                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.device)
                batch_nodes, batch_edge_idx, batch_idx = batch.x, batch.edge_index, batch.batch
                batch_nodes = F.normalize(batch_nodes, p=2, dim=-1)

                batch_reverse = Batch.from_data_list(data_list_reverse)
                batch_reverse = batch_reverse.to(self.device)
                batch_nodes_reverse, batch_edge_idx_reverse, batch_idx_reverse = batch_reverse.x, \
                                                                                 batch_reverse.edge_index, \
                                                                                 batch_reverse.batch
                batch_nodes_reverse = F.normalize(batch_nodes_reverse, p=2, dim=-1)

                pred = self.stage1(batch_nodes, batch_edge_idx, batch_idx,
                                   batch_nodes_reverse, batch_edge_idx_reverse).squeeze()
                if len(pred.size()) == 0:
                    pred.unsqueeze_(0)
                pred_list.append(pred)
        return torch.cat(pred_list, dim=0)

    def fit_g(self, ytrain, arch_data_edge_idx_list, arch_data_node_f_list, arch_data_edge_idx_reverse_list,
              arch_data_node_f_reverse_list, ytest, data_list, data_list_reverse):
        start = time.time()
        self.fit(arch_data_edge_idx_list, arch_data_node_f_list, arch_data_edge_idx_reverse_list,
                 arch_data_node_f_reverse_list, ytrain)
        duration = time.time() - start
        predictions = self.pred(data_list, data_list_reverse)
        predictions = predictions.cpu().numpy()
        s_t = get_spearmanr_coorlection(predictions, ytest)[0]
        k_t = get_kendalltau_coorlection(predictions, ytest)[0]
        if self.logger:
            self.logger.info(f'Training time cost: {duration}, Spearman Correlation: {s_t}, '
                             f'Kendalltau Corrlation: {k_t}')
        else:
            print(f'Training time cost: {duration}, '
                  f'Spearman Correlation: {s_t}, '
                  f'Kendalltau Corrlation: {k_t}')
        return s_t, k_t, duration