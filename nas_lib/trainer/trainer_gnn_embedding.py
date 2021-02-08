# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved


from nas_lib.predictors.predictor_unsupervised_siamese_ged import PredictorSiameseGED
from nas_lib.utils.utils_data import nas2graph, get_node_num, get_ops_list
import torch
from ..utils.utils_solver import CosineLR, gen_batch_idx_gen, make_agent_optimizer
from ..utils.metric_logger import MetricLogger
import random
from gnn_lib.data import Data, Batch
from functools import partial
from nas_lib.utils.utils_data import edit_distance, edit_distance_normalization
import os
import numpy as np
from nas_lib.predictors_compare.BRP_NAS.utils import ProductList


class TrainerGED:
    def __init__(self, gpu, nas_benchmark, predictor_type, lr, epochs, batch_size, input_dim, epoch_img_size,
                 model_save_dir, ratio=None, full_train=False, save_model=True, logger=None, args=None):
        self.gpu = gpu
        self.nas_benchmark = nas_benchmark
        self.device = torch.device(f'cuda:{gpu}')
        self.predictor_type = predictor_type
        self.input_dim = input_dim
        self.node_num = get_node_num(nas_benchmark)
        self.node_type_num = get_node_num(nas_benchmark)
        self.ops_list = get_ops_list(nas_benchmark)
        self.predictor = PredictorSiameseGED(input_dim=input_dim)
        self.optimizer = make_agent_optimizer(self.predictor, base_lr=lr, weight_deacy=1e-4, bias_multiply=True)
        self.predictor.to(self.device)
        self.lr = lr
        self.scheduler = CosineLR(self.optimizer, epochs=epochs, train_images=epoch_img_size, batch_size=batch_size)
        self.batch_size = batch_size
        self.epoch = epochs
        self.nas2graph_p = partial(nas2graph, nas_benchmark)
        self.save_dir = model_save_dir
        self.criterion = torch.nn.MSELoss()
        self.full_train = full_train
        self.save_model = save_model
        self.logger = logger
        self.args = args
        self.ratio = ratio

    def fit(self, total_archs):
        train_g_data, train_encoding = self._split_dataset(total_archs)

        for e in range(self.epoch):
            if self.logger:
                self.logger.info(f'epoch {e} begin')
            else:
                print(f'epoch {e} begin')

            if self.full_train:
                self.fit_full_train(train_g_data, train_encoding, e)
            else:
                self.fit_train(train_g_data, train_encoding, e)

    def fit_train(self, train_g_data_list, arch_encoding, epoch):
        meters = MetricLogger(delimiter=" ")
        self.predictor.train()
        idx_list = list(range(len(train_g_data_list)))
        idx_list2 = list(idx_list)
        random.shuffle(idx_list)
        random.shuffle(idx_list2)
        if self.args and self.args.add_corresponding:
            idx = list(range(len(train_g_data_list)))
            idx2 = list(range(len(train_g_data_list)))
            idx_list.extend(idx)
            idx_list2.extend(idx2)
            indices_list = list(range(len(idx_list)))
            random.shuffle(indices_list)
            idx_list = [idx_list[i] for i in indices_list]
            idx_list2 = [idx_list2[i] for i in indices_list]
        batch_idx_list_1 = gen_batch_idx_gen(idx_list, self.batch_size, drop_last=True)
        batch_idx_list_2 = gen_batch_idx_gen(idx_list2, self.batch_size, drop_last=True)
        counter = 0
        for i, (pair1_idx, pair2_idx) in enumerate(zip(batch_idx_list_1, batch_idx_list_2)):
            counter += len(pair1_idx)
            data_list_pair1 = []
            arch_path_encoding_pair1 = []

            data_list_pair2 = []
            arch_path_encoding_pair2 = []

            for pair_idx in zip(pair1_idx, pair2_idx):
                idx1, idx2 = pair_idx
                g_d_1 = train_g_data_list[idx1]
                data_list_pair1.append(g_d_1)
                arch_path_encoding_pair1.append(arch_encoding[idx1])

                g_d_2 = train_g_data_list[idx2]
                data_list_pair2.append(g_d_2)
                arch_path_encoding_pair2.append(arch_encoding[idx2])

            if self.args.ged_type == 'normalized':
                dist_gt = torch.tensor([edit_distance_normalization(arch_path_encoding_pair1[i],
                                                                    arch_path_encoding_pair2[i], self.node_num)
                                        for i in range(len(arch_path_encoding_pair1))], dtype=torch.float32)
            elif self.args.ged_type == 'wo_normalized':
                dist_gt = torch.tensor([edit_distance(arch_path_encoding_pair1[i],
                                                      arch_path_encoding_pair2[i])
                                        for i in range(len(arch_path_encoding_pair1))], dtype=torch.float32)
            else:
                raise ValueError(f'The ged type {self.args.ged_type} does not support!')
            batch1 = Batch.from_data_list(data_list_pair1)
            batch1 = batch1.to(self.device)

            batch2 = Batch.from_data_list(data_list_pair2)
            batch2 = batch2.to(self.device)

            dist_gt = dist_gt.to(self.device)

            batch_nodes_1, batch_edge_idx_1, batch_idx_1 = batch1.x, batch1.edge_index, batch1.batch
            batch_nodes_2, batch_edge_idx_2, batch_idx_2 = batch2.x, batch2.edge_index, batch2.batch
            prediction = self.predictor(batch_nodes_1, batch_edge_idx_1, batch_idx_1, batch_nodes_2,
                                        batch_edge_idx_2, batch_idx_2)
            prediction = prediction.squeeze(dim=-1)
            loss = self.criterion(prediction, dist_gt)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            meters.update(loss=loss.item())
        save_dir = os.path.join(self.save_dir, f'unsupervised_ss_rl_epoch_{epoch}.pt')
        if self.save_model:
            torch.save(self.predictor.state_dict(), save_dir)
        if self.logger:
            self.logger.info(meters.delimiter.join(['{loss}'.format(loss=str(meters))]))
        else:
            print(meters.delimiter.join(['{loss}'.format(loss=str(meters))]))

    def fit_full_train(self, train_g_data_list, arch_encoding, epoch):
        meters = MetricLogger(delimiter=" ")
        self.predictor.train()
        idx_list = list(range(len(train_g_data_list)))
        idx_dataset = ProductList(idx_list)
        training_data = torch.utils.data.DataLoader(idx_dataset, batch_size=self.batch_size,
                                                    shuffle=True)
        counter = 0
        for i, v in enumerate(training_data):
            pair1_idx = v[0].cpu().numpy()
            pair2_idx = v[1].cpu().numpy()

            counter += len(pair1_idx)
            data_list_pair1 = []
            arch_path_encoding_pair1 = []

            data_list_pair2 = []
            arch_path_encoding_pair2 = []

            for pair_idx in zip(pair1_idx, pair2_idx):
                idx1, idx2 = pair_idx
                g_d_1 = train_g_data_list[idx1]
                data_list_pair1.append(g_d_1)
                arch_path_encoding_pair1.append(arch_encoding[idx1])

                g_d_2 = train_g_data_list[idx2]
                data_list_pair2.append(g_d_2)
                arch_path_encoding_pair2.append(arch_encoding[idx2])

            dist_gt = torch.tensor([edit_distance_normalization(arch_path_encoding_pair1[i],
                                                                arch_path_encoding_pair2[i], self.node_num)
                                    for i in range(len(arch_path_encoding_pair1))], dtype=torch.float32)
            batch1 = Batch.from_data_list(data_list_pair1)
            batch1 = batch1.to(self.device)

            batch2 = Batch.from_data_list(data_list_pair2)
            batch2 = batch2.to(self.device)

            dist_gt = dist_gt.to(self.device)

            batch_nodes_1, batch_edge_idx_1, batch_idx_1 = batch1.x, batch1.edge_index, batch1.batch
            batch_nodes_2, batch_edge_idx_2, batch_idx_2 = batch2.x, batch2.edge_index, batch2.batch
            prediction = self.predictor(batch_nodes_1, batch_edge_idx_1, batch_idx_1, batch_nodes_2,
                                        batch_edge_idx_2, batch_idx_2)
            prediction = prediction.squeeze(dim=-1)
            loss = self.criterion(prediction, dist_gt)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            meters.update(loss=loss.item())
        save_dir = os.path.join(self.save_dir, f'unsupervised_ss_rl_epoch_{epoch}.pt')
        if self.save_model:
            torch.save(self.predictor.state_dict(), save_dir)
        if self.logger:
            self.logger.info(meters.delimiter.join(['{loss}'.format(loss=str(meters))]))
        else:
            print(meters.delimiter.join(['{loss}'.format(loss=str(meters))]))

    def inference(self, edge_index, node_feature, arch_encoding, batch_idx_test_1, batch_idx_test_2):
        self.predictor.eval()
        error_list = []
        precision_list = []
        for i, pair1_idx in enumerate(batch_idx_test_1):
            pair2_idx = batch_idx_test_2[i]
            data_list_pair1 = []
            arch_path_encoding_pair1 = []

            data_list_pair2 = []
            arch_path_encoding_pair2 = []

            for pair_idx in zip(pair1_idx, pair2_idx):
                idx1, idx2 = pair_idx
                g_d_1 = Data(edge_index=edge_index[idx1].long(), x=node_feature[idx1].float())
                data_list_pair1.append(g_d_1)
                arch_path_encoding_pair1.append(arch_encoding[idx1])

                g_d_2 = Data(edge_index=edge_index[idx2].long(), x=node_feature[idx2].float())
                data_list_pair2.append(g_d_2)
                arch_path_encoding_pair2.append(arch_encoding[idx2])

            dist_gt = torch.tensor([edit_distance(arch_path_encoding_pair1[i], arch_path_encoding_pair2[i])
                                    for i in range(len(arch_path_encoding_pair1))], dtype=torch.float32)
            batch1 = Batch.from_data_list(data_list_pair1)
            batch1 = batch1.to(self.device)

            batch2 = Batch.from_data_list(data_list_pair2)
            batch2 = batch2.to(self.device)

            dist_gt = dist_gt.to(self.device)

            batch_nodes_1, batch_edge_idx_1, batch_idx_1 = batch1.x, batch1.edge_index, batch1.batch
            batch_nodes_2, batch_edge_idx_2, batch_idx_2 = batch2.x, batch2.edge_index, batch2.batch
            prediction = self.predictor(batch_nodes_1, batch_edge_idx_1, batch_idx_1, batch_nodes_2,
                                        batch_edge_idx_2, batch_idx_2)
            prediction = -1 * torch.log(prediction.squeeze(dim=-1)) * self.node_num

            errors = torch.abs(dist_gt - prediction)
            precision = (torch.sum(errors < 1) * 1.) / errors.size(0)
            error_list.append(torch.mean(errors).item())
            precision_list.append(precision.item())
        if self.logger:
            self.logger.info(f'Error is {np.mean(np.array(error_list))}, Precision is {np.mean(np.array(precision_list))}')
        else:
            print(f'Error is {np.mean(np.array(error_list))}, Precision is {np.mean(np.array(precision_list))}')

    def _verify_node_nums(self, node_feature_list):
        node_nums = node_feature_list[0].shape[0]
        for node in node_feature_list:
            if node.shape[0] != node_nums:
                print(node)
                raise ValueError('The node num is incorrect!!!')

    def _split_dataset(self, total_archs):
        if self.ratio:
            train_nums = int(self.ratio * len(total_archs))
            train_arch_g_data_list = [arch[0] for arch in total_archs[:train_nums]]
            train_path_encoding = [arch[1] for arch in total_archs[:train_nums]]
        else:
            train_arch_g_data_list = [arch[0] for arch in total_archs]
            train_path_encoding = [arch[1] for arch in total_archs]
        return train_arch_g_data_list, train_path_encoding