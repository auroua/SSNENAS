# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved

import torch
import random
from nas_lib.predictors.predictor_gin import PredictorGIN
from gnn_lib.data import Data, Batch
from ..utils.metric_logger import MetricLogger
from ..utils.utils_solver import CosineLR, gen_batch_idx, make_agent_optimizer
from nas_lib.utils.utils_model import get_temp_model, load_modify_model, load_predictor_ged_moco_v2
from nas_lib.predictor_retrain_compare.predictor_gin_rl import PredictorGINRL
from nas_lib.predictor_retrain_compare.predictor_gin_ccl import PredictorGINCCL


class NasBenchGinPredictorTrainer:
    def __init__(self, agent_type, lr=0.01, device=None, epochs=10, train_images=10, batch_size=10, rate=10,
                 input_dim=6, model_dir=None, predictor_type=None, logger=None, algo_name=None):
        if algo_name and 'ss' in algo_name:
            self.nas_agent = self._get_predictor(algo_name)(input_dim=input_dim, reTrain=True)
        else:
            self.nas_agent = PredictorGIN(input_dim=input_dim)
        self.lr = lr

        if model_dir:
            model_path = self._get_model_dir(algo_name, model_dir)
            if 'ss_rl' in algo_name:
                temp_model = get_temp_model(predictor_type, input_dim)
                self.nas_agent = load_modify_model(self.nas_agent, temp_model, model_path)
                del temp_model
                self.lr *= 0.1
                self.nas_agent.fc = torch.nn.Linear(32, 1, bias=True)
                torch.nn.init.kaiming_uniform_(self.nas_agent.fc.weight, a=1)
                self.nas_agent.fc.bias.data.zero_()

            elif 'ss_ccl' in algo_name:
                self.nas_agent = load_predictor_ged_moco_v2(self.nas_agent, model_path)
                # self.lr *= 0.1
                self.nas_agent.fc = torch.nn.Linear(32, 1, bias=True)
                torch.nn.init.kaiming_uniform_(self.nas_agent.fc.weight, a=1)
                self.nas_agent.fc.bias.data.zero_()

            logger.info(f'Predictor {predictor_type} successfully loaded!')
        self.agent_type = agent_type

        self.criterion = torch.nn.MSELoss()
        self.optimizer = make_agent_optimizer(self.nas_agent, base_lr=lr, weight_deacy=1e-4, bias_multiply=True)
        self.device = device
        self.nas_agent.to(self.device)
        self.scheduler = CosineLR(self.optimizer, epochs=epochs, train_images=train_images, batch_size=batch_size)
        self.batch_size = batch_size
        self.epoch = epochs
        self.rate = rate

    def fit(self, edge_index, node_feature, val_accuracy, logger=None):
        meters = MetricLogger(delimiter=" ")
        self.nas_agent.train()
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
                    target_list.append(val_accuracy[idx])
                val_tensor = torch.tensor(target_list, dtype=torch.float32)
                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.device)
                val_tensor = val_tensor.to(self.device)
                batch_nodes, batch_edge_idx, batch_idx = batch.x, batch.edge_index, batch.batch
                # batch_nodes = F.normalize(batch_nodes, p=2, dim=-1)

                pred = self.nas_agent(batch_nodes, batch_edge_idx, batch_idx)*self.rate
                pred = pred.squeeze()
                loss = self.criterion(pred, val_tensor)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step(epoch + int(i/30))
                self.scheduler.step()
                meters.update(loss=loss.item())
        if logger:
            logger.info(meters.delimiter.join(['{loss}'.format(loss=str(meters))]))
        return meters.meters['loss'].avg

    def pred(self, edge_index, node_feature):
        pred_list = []
        idx_list = list(range(len(edge_index)))
        self.nas_agent.eval()
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
                pred = self.nas_agent(batch_nodes, batch_edge_idx, batch_idx).squeeze()*self.rate

                if len(pred.size()) == 0:
                    pred.unsqueeze_(0)
                pred_list.append(pred)
        return torch.cat(pred_list, dim=0)

    def _get_predictor(self, algo_name):
        if 'ss_rl' in algo_name:
            return PredictorGINRL
        elif 'ss_ccl' in algo_name:
            return PredictorGINCCL
        else:
            raise NotImplementedError(f'In module trainer_predictor the {algo_name} does not support!')

    def _get_model_dir(self, algo_name, model_dir_dict):
        for k, v in model_dir_dict.items():
            if k in algo_name:
                return v
        raise ValueError(f'The algo name {algo_name} does not have a pre-trained ckpt file.')
