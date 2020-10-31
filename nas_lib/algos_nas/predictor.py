# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved
import numpy as np
import torch
import copy
from nas_lib.utils.utils_data import nasbench2graph_101, nasbench2graph_201
from nas_lib.eigen.trainer_predictor import NasBenchGinPredictorTrainer


def gin_predictor_nasbench_101(search_space,
                               num_init=10,
                               k=10,
                               total_queries=150,
                               acq_opt_type='mutation',
                               allow_isomorphisms=False,
                               verbose=1,
                               agent=None,
                               logger=None,
                               gpu='0',
                               lr=0.01,
                               candidate_nums=100,
                               epochs=1000,
                               algo_name=None):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    if len(data) <= 10:
        batch_size = 10
    else:
        batch_size = 16
    while query <= total_queries:
        arch_data = [d[0] for d in data]
        agent = NasBenchGinPredictorTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                            train_images=len(data), batch_size=batch_size, algo_name=algo_name)
        val_accuracy = np.array([d[4] for d in data])
        arch_data_edge_idx_list = []
        arch_data_node_f_list = []
        for arch in arch_data:
            edge_index, node_f = nasbench2graph_101(arch)
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)
        candidates = search_space.get_candidates(data,
                                                 num=candidate_nums,
                                                 acq_opt_type=acq_opt_type,
                                                 allow_isomorphisms=allow_isomorphisms)
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = nasbench2graph_101(cand[0])
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
        acc_train = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        acc_pred = agent.pred(candiate_edge_list, candiate_node_list)
        candidate_np = acc_pred.cpu().numpy()
        sorted_indices = np.argsort(candidate_np)
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch(matrix=candidates[i][1],
                                                ops=candidates[i][2])
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training mean loss is  {}'.format(query,
                                                                     np.mean(np.abs(acc_train.cpu().numpy()-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data


def gin_predictor_nasbench_201(search_space,
                               num_init=10,
                               k=10,
                               total_queries=150,
                               acq_opt_type='mutation',
                               allow_isomorphisms=False,
                               verbose=1,
                               agent=None,
                               logger=None,
                               gpu='0',
                               lr=0.01,
                               candidate_nums=100,
                               epochs=1000,
                               rate=10,
                               algo_name=None):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    if len(data) <= 10:
        batch_size = 10
    else:
        batch_size = 16
    while query <= total_queries:
        arch_data = [d[0] for d in data]
        agent = NasBenchGinPredictorTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                            train_images=len(data), batch_size=batch_size, input_dim=8, rate=rate,
                                            algo_name=algo_name)
        val_accuracy = np.array([d[4] for d in data])
        arch_data_edge_idx_list = []
        arch_data_node_f_list = []
        for arch in arch_data:
            edge_index, node_f = nasbench2graph_201(arch)
            arch_data_edge_idx_list.append(edge_index)
            arch_data_node_f_list.append(node_f)
        candidates = search_space.get_candidates(data,
                                                 num=candidate_nums,
                                                 allow_isomorphisms=allow_isomorphisms)
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = nasbench2graph_201(cand[0])
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
        acc_train = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list)
        acc_pred = agent.pred(candiate_edge_list, candiate_node_list)
        candidate_np = acc_pred.cpu().numpy()
        sorted_indices = np.argsort(candidate_np)
        for i in sorted_indices[:k]:
            archtuple = candidates[i]
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training mean loss is  {}'.format(query,
                                                                     np.mean(np.abs(acc_train.cpu().numpy()-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
    return data


def gin_predictor_train_num_restract_nasbench_101(search_space,
                                                  num_init=10,
                                                  k=10,
                                                  total_queries=150,
                                                  acq_opt_type='mutation',
                                                  allow_isomorphisms=False,
                                                  verbose=1,
                                                  agent=None,
                                                  logger=None,
                                                  gpu='0',
                                                  lr=0.01,
                                                  candidate_nums=100,
                                                  epochs=1000,
                                                  training_nums=150,
                                                  algo_name=None):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    train_data = []
    train_flag = False
    while query <= total_queries:
        if len(train_data) < training_nums:
            train_data = copy.deepcopy(data)
            train_flag = True
        batch_size = 10 if len(train_data) <= 10 else 16
        candidates = search_space.get_candidates(data,
                                                 num=candidate_nums,
                                                 acq_opt_type=acq_opt_type,
                                                 allow_isomorphisms=allow_isomorphisms)
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = nasbench2graph_101(cand[0])
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        if train_flag:
            arch_data = [d[0] for d in train_data]
            val_accuracy = np.array([d[4] for d in train_data])
            agent = NasBenchGinPredictorTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                                train_images=len(train_data), batch_size=batch_size,
                                                algo_name=algo_name)
            arch_data_edge_idx_list = []
            arch_data_node_f_list = []
            for arch in arch_data:
                edge_index, node_f = nasbench2graph_101(arch)
                arch_data_edge_idx_list.append(edge_index)
                arch_data_node_f_list.append(node_f)
            agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
            acc_train = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list).cpu().numpy()
        acc_pred = agent.pred(candiate_edge_list, candiate_node_list)
        candidate_np = acc_pred.cpu().numpy()
        sorted_indices = np.argsort(candidate_np)
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch(matrix=candidates[i][1],
                                                ops=candidates[i][2])
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training data nums {},  training mean loss is  {}'.format(query, len(train_data),
                                                                     np.mean(np.abs(acc_train-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
        train_flag = False
    return data


def gin_predictor_train_num_restract_nasbench_201(search_space,
                                                  num_init=10,
                                                  k=10,
                                                  total_queries=150,
                                                  acq_opt_type='mutation',
                                                  allow_isomorphisms=False,
                                                  verbose=1,
                                                  agent=None,
                                                  logger=None,
                                                  gpu='0',
                                                  lr=0.01,
                                                  candidate_nums=100,
                                                  epochs=1000,
                                                  training_nums=150,
                                                  rate=10,
                                                  algo_name=None):
    """
    Bayesian optimization with a neural network model
    """
    device = torch.device('cuda:%d' % gpu)
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=True)
    query = num_init + k
    search_agent = agent
    train_data = []
    train_flag = False
    while query <= total_queries:
        if len(train_data) < training_nums:
            train_data = copy.deepcopy(data)
            train_flag = True
        batch_size = 10 if len(train_data) <= 10 else 16
        candidates = search_space.get_candidates(data,
                                                 num=candidate_nums,
                                                 allow_isomorphisms=allow_isomorphisms)
        candiate_edge_list = []
        candiate_node_list = []
        for cand in candidates:
            edge_index, node_f = nasbench2graph_201(cand[0])
            candiate_edge_list.append(edge_index)
            candiate_node_list.append(node_f)
        if train_flag:
            arch_data = [d[0] for d in train_data]
            val_accuracy = np.array([d[4] for d in train_data])
            agent = NasBenchGinPredictorTrainer(search_agent, lr=lr, device=device, epochs=epochs,
                                                train_images=len(data), batch_size=batch_size, input_dim=8, rate=rate,
                                                algo_name=algo_name)
            arch_data_edge_idx_list = []
            arch_data_node_f_list = []
            for arch in arch_data:
                edge_index, node_f = nasbench2graph_201(arch)
                arch_data_edge_idx_list.append(edge_index)
                arch_data_node_f_list.append(node_f)
            agent.fit(arch_data_edge_idx_list, arch_data_node_f_list, val_accuracy, logger=None)
            acc_train = agent.pred(arch_data_edge_idx_list, arch_data_node_f_list).cpu().numpy()
        acc_pred = agent.pred(candiate_edge_list, candiate_node_list)
        candidate_np = acc_pred.cpu().numpy()
        sorted_indices = np.argsort(candidate_np)
        for i in sorted_indices[:k]:
            archtuple = candidates[i]
            data.append(archtuple)
        if verbose:
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, training data nums {},  training mean loss is  {}'.format(query, len(train_data),
                                                                     np.mean(np.abs(acc_train-val_accuracy))))
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += k
        train_flag = False
    return data