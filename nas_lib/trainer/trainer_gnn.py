# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved

from nas_lib.predictor_retrain_compare.predictor_gin_rl import PredictorGINRL
from nas_lib.predictor_retrain_compare.predictor_gin_ccl import PredictorGINCCL
from nas_lib.utils.utils_data import nas2graph
import torch
from ..utils.utils_solver import CosineLR, gen_batch_idx, make_agent_optimizer
from ..utils.metric_logger import MetricLogger
import random
from gnn_lib.data import Data, Batch
from functools import partial
import numpy as np
from nas_lib.utils.comm import get_spearmanr_coorlection, get_kendalltau_coorlection
from nas_lib.utils.utils_model import get_temp_model, load_modify_model, \
    load_predictor_ged_moco_v2
import os
import time


class TrainerGNN:
    def __init__(self, gpu, nas_benchmark, predictor_type, lr, epochs, batch_size, input_dim, epoch_img_size,
                 load_model, model_path, save_path, save_model=False, with_g_func=True, logger=None):
        self.gpu = gpu
        self.nas_benchmark = nas_benchmark
        self.device = torch.device(f'cuda:{gpu}')
        self.predictor_type = predictor_type
        self.input_dim = input_dim
        self.criterion = torch.nn.MSELoss()
        self.predictor = self._get_predicotr(with_g_func)
        self.lr = lr
        if load_model and predictor_type == 'SS_RL':
            temp_model = get_temp_model(predictor_type, input_dim)
            self.predictor = load_modify_model(self.predictor, temp_model, model_path)
            logger.info(f'load model {predictor_type}')
            del temp_model
            self.lr *= 0.1
        elif load_model and 'SS_CCL' in predictor_type:
            self.predictor = load_predictor_ged_moco_v2(self.predictor, model_path)
            logger.info(f'load model {predictor_type}')
        if 'SS_CCL' in predictor_type:
            self.predictor.fc = torch.nn.Linear(32, 1, bias=True)
            torch.nn.init.kaiming_uniform_(self.predictor.fc.weight, a=1)
            self.predictor.fc.bias.data.zero_()
        self.optimizer = make_agent_optimizer(self.predictor, base_lr=lr, weight_deacy=1e-4, bias_multiply=True)
        self.predictor.to(self.device)

        self.scheduler = CosineLR(self.optimizer, epochs=epochs, train_images=epoch_img_size, batch_size=batch_size)
        self.batch_size = batch_size
        self.epoch = epochs
        self.nas2graph_p = partial(nas2graph, nas_benchmark)
        self.save_dir = save_path
        self.save_model = save_model
        self.logger = logger

    def fit(self, train_archs, ytrain, test_archs, ytest):
        arch_data_edge_idx_list = []
        arch_data_node_f_list = []

        for arch in train_archs:
            edge_index, node_f = self.nas2graph_p(arch)
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)

        candiate_edge_list = []
        candiate_node_list = []
        for cand in test_archs:
            edge_index, node_f = self.nas2graph_p(cand)
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)

        self.fit_train(arch_data_edge_idx_list, arch_data_node_f_list, ytrain)

        train_pred = self.pred(arch_data_edge_idx_list, arch_data_node_f_list).cpu().numpy()
        test_pred = self.pred(candiate_edge_list, candiate_node_list).cpu().numpy()

        train_mean_error = np.mean(np.abs(train_pred - ytrain))
        test_mean_error = np.mean(np.abs(test_pred - ytest))

        test_spearman_corr = get_spearmanr_coorlection(test_pred.tolist(), ytest)[0]
        test_kendalltau_corr = get_kendalltau_coorlection(test_pred.tolist(), ytest)[0]

        if self.logger:
            self.logger.info(f'Train error: {train_mean_error}, Test error: {test_mean_error}, Spearman Correlation: {test_spearman_corr}, '
                             f'Kendalltau Corrlation: {test_kendalltau_corr}')
        else:
            print(f'Train error: {train_mean_error}, Test error: {test_mean_error}, Spearman Correlation: {test_spearman_corr}, '
                  f'Kendalltau Corrlation: {test_kendalltau_corr}')
        return test_spearman_corr, test_kendalltau_corr

    def fit_train(self, edge_index, node_feature, accuracy, logger=None):
        meters = MetricLogger(delimiter=" ")
        self.predictor.train()
        start = time.time()
        for epoch in range(self.epoch):
            idx_list = list(range(len(edge_index)))
            random.shuffle(idx_list)
            batch_idx_list = gen_batch_idx(idx_list, self.batch_size)
            counter = 0
            for i, batch_idx in enumerate(batch_idx_list):
                counter += len(batch_idx)
                data_list = []
                target_list = []
                for idx in batch_idx:
                    g_d = Data(edge_index=edge_index[idx].long(), x=node_feature[idx].float())
                    data_list.append(g_d)
                    target_list.append(accuracy[idx])
                val_tensor = torch.tensor(target_list, dtype=torch.float32)
                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.device)
                val_tensor = val_tensor.to(self.device)
                batch_nodes, batch_edge_idx, batch_idx = batch.x, batch.edge_index, batch.batch
                # batch_nodes = F.normalize(batch_nodes, p=2, dim=-1)

                pred = self.predictor(batch_nodes, batch_edge_idx, batch_idx)
                pred = pred.squeeze()
                loss = self.criterion(pred, val_tensor)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                meters.update(loss=loss.item())
            # print(meters.delimiter.join(['{loss}'.format(loss=str(meters))]))
            save_dir = os.path.join(self.save_dir, f'supervised_gin_epoch_{epoch}.pt')
            if self.save_model:
                torch.save(self.predictor.state_dict(), save_dir)
        return meters.meters['loss'].avg

    def pred(self, edge_index, node_feature):
        pred_list = []
        idx_list = list(range(len(edge_index)))
        self.predictor.eval()
        batch_idx_list = gen_batch_idx(idx_list, 64)
        with torch.no_grad():
            for batch_idx in batch_idx_list:
                data_list = []
                for idx in batch_idx:
                    g_d = Data(edge_index=edge_index[idx].long(), x=node_feature[idx].float())
                    data_list.append(g_d)
                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.device)
                batch_nodes, batch_edge_idx, batch_idx = batch.x, batch.edge_index, batch.batch
                pred = self.predictor(batch_nodes, batch_edge_idx, batch_idx).squeeze()

                if len(pred.size()) == 0:
                    pred.unsqueeze_(0)
                pred_list.append(pred)
        return torch.cat(pred_list, dim=0)

    def fit_g(self, ytrain, g_train_data, ytest, g_test_data):
        start = time.time()
        self.fit_train_g_data(g_train_data, ytrain)
        duration = time.time() - start
        train_pred = self.pred_g_data(g_train_data).cpu().numpy()
        test_pred = self.pred_g_data(g_test_data).cpu().numpy()

        train_mean_error = np.mean(np.abs(train_pred - ytrain))
        test_mean_error = np.mean(np.abs(test_pred - ytest))

        test_spearman_corr = get_spearmanr_coorlection(test_pred.tolist(), ytest)[0]
        test_kendalltau_corr = get_kendalltau_coorlection(test_pred.tolist(), ytest)[0]

        if self.logger:
            self.logger.info(f'Training time cost: {duration}, Train error: {train_mean_error}, '
                             f'Test error: {test_mean_error}, Spearman Correlation: {test_spearman_corr}, '
                             f'Kendalltau Corrlation: {test_kendalltau_corr}')
        else:
            print(f'Training time cost: {duration}, Train error: {train_mean_error}, Test error: {test_mean_error}, '
                  f'Spearman Correlation: {test_spearman_corr}, '
                  f'Kendalltau Corrlation: {test_kendalltau_corr}')
        return test_spearman_corr, test_kendalltau_corr, duration

    def fit_train_g_data(self, g_data, accuracy, logger=None):
        meters = MetricLogger(delimiter=" ")
        self.predictor.train()
        for epoch in range(self.epoch):
            idx_list = list(range(len(g_data)))
            random.shuffle(idx_list)
            batch_idx_list = gen_batch_idx(idx_list, self.batch_size)
            counter = 0
            for i, batch_idx in enumerate(batch_idx_list):
                counter += len(batch_idx)
                data_list = [g_data[id] for id in batch_idx]
                target_list = [accuracy[id] for id in batch_idx]
                val_tensor = torch.tensor(target_list, dtype=torch.float32)
                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.device)
                val_tensor = val_tensor.to(self.device)
                batch_nodes, batch_edge_idx, batch_idx = batch.x, batch.edge_index, batch.batch
                # batch_nodes = F.normalize(batch_nodes, p=2, dim=-1)

                pred = self.predictor(batch_nodes, batch_edge_idx, batch_idx)
                pred = pred.squeeze()
                loss = self.criterion(pred, val_tensor)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                meters.update(loss=loss.item())
            save_dir = os.path.join(self.save_dir, f'supervised_gin_epoch_{epoch}.pt')
            if self.save_model:
                torch.save(self.predictor.state_dict(), save_dir)
        return meters.meters['loss'].avg

    def pred_g_data(self, g_data):
        pred_list = []
        idx_list = list(range(len(g_data)))
        self.predictor.eval()
        batch_idx_list = gen_batch_idx(idx_list, 64)
        with torch.no_grad():
            for batch_idx in batch_idx_list:
                data_list = [g_data[idx] for idx in batch_idx]
                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.device)
                batch_nodes, batch_edge_idx, batch_idx = batch.x, batch.edge_index, batch.batch
                pred = self.predictor(batch_nodes, batch_edge_idx, batch_idx).squeeze()
                if len(pred.size()) == 0:
                    pred.unsqueeze_(0)
                pred_list.append(pred)
        return torch.cat(pred_list, dim=0)

    def _get_predicotr(self, with_g_func):
        if self.predictor_type == 'SS_RL' or 'SS_RL' in self.predictor_type:
            predictor = PredictorGINRL(input_dim=self.input_dim)
        elif 'SS_CCL' in self.predictor_type:
            predictor = PredictorGINCCL(input_dim=self.input_dim, reTrain=True)
        else:
            raise NotImplementedError(f'The predictor type {self.predictor_type} have implement yet!')
        return predictor