# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved


from nas_lib.params import get_params
from nas_lib.trainer.trainer_gnn import TrainerGNN
from nas_lib.eigen_predictive_compare.semi_nas_trainer import semi_nas_trainer
from nas_lib.eigen_predictive_compare.np_nas_trainer import NasBenchGcnNnpTrainer
from nas_lib.eigen_predictive_compare.mlp_trainer import MetaNeuralnetTrainer
import math


END_NUM = -1
# END_NUM = 42362


class NASBenchReTrain:
    def __init__(self, args, predictor_type, train_size, load_model=False, load_dir=None, train_epochs=None,
                 logger=None):
        params = get_params(args, predictor_type, load_model=load_model)
        # if train_size == 200 and load_model and predictor_type == 'SS_CCL':
        #     params['lr'] = 0.1*params['lr']
        self.trainer = build_predictor(predictor_type=predictor_type,
                                       params=params,
                                       args=args,
                                       train_size=train_size,
                                       load_model=load_model,
                                       load_dir=load_dir,
                                       train_epochs=train_epochs,
                                       logger=logger)
        self.predictor_type = predictor_type
        self.params = params
        self.args = args

    def fit(self, train_data, test_data):
        train_archs = [(d['matrix'], d['ops']) for d in train_data]
        ytrain = [d['val_acc'] for d in train_data]
        test_matrix = [(d['matrix'], d['ops']) for d in test_data]
        ytest = [d['val_acc'] for d in test_data]
        spearman_corr, kendalltau_corr = self.trainer.fit(train_archs, ytrain, test_matrix, ytest)
        return spearman_corr, kendalltau_corr

    def fit_g_data(self, train_data, test_data):
        # train_archs = [(d['matrix'], d['ops']) for d in train_data]
        ytrain = [d['val_acc'] for d in train_data]
        g_train_data = [d['g_data'] for d in train_data]
        # test_archs = [(d['matrix'], d['ops']) for d in test_data][:END_NUM]
        ytest = [d['val_acc'] for d in test_data][:END_NUM]
        g_test_data = [d['g_data'] for d in test_data][:END_NUM]
        if self.predictor_type == 'SemiNAS':
            train_semi_vecs = [d['seminas_vec'] for d in train_data]
            test_semi_vecs = [d['seminas_vec'] for d in test_data][:END_NUM]
            spearman_corr, kendalltau_corr, duration = self.trainer.fit_g(ytrain, train_semi_vecs, ytest, test_semi_vecs)
        elif self.predictor_type == 'BANANAS_ADJ' or self.predictor_type == 'MLP':
            train_data = [d['pe_adj_enc_vec'] for d in train_data]
            test_data = [d['pe_adj_enc_vec'] for d in test_data][:END_NUM]
            spearman_corr, kendalltau_corr, duration = self.trainer.fit_g(ytrain, train_data, ytest, test_data)
        elif self.predictor_type == 'NP_NAS':
            arch_data_edge_idx_list = [d['edge_idx'] for d in train_data]
            arch_data_node_f_list = [d['node_f'] for d in train_data]
            arch_data_edge_idx_reverse_list = [d['edge_idx_reverse'] for d in train_data]
            arch_data_node_f_reverse_list = [d['node_f_reverse'] for d in train_data]

            candidate_g_data = [d['g_data'] for d in test_data][:END_NUM]
            candidate_g_reverse_data = [d['g_data_reverse'] for d in test_data][:END_NUM]
            spearman_corr, kendalltau_corr, duration = self.trainer.fit_g(ytrain, arch_data_edge_idx_list,
                                                                          arch_data_node_f_list,
                                                                          arch_data_edge_idx_reverse_list,
                                                                          arch_data_node_f_reverse_list,
                                                                          ytest, candidate_g_data,
                                                                          candidate_g_reverse_data)
        else:
            spearman_corr, kendalltau_corr, duration = self.trainer.fit_g(ytrain, g_train_data, ytest, g_test_data)
        return spearman_corr, kendalltau_corr, duration


def build_predictor(predictor_type, train_size, params, args, load_model, load_dir, train_epochs, logger):
    if predictor_type == 'NP_NAS':
        trainer = NasBenchGcnNnpTrainer(
            device=args.gpu,
            epochs=params['epochs'],
            input_dim=params['input_dim'],
            params=params,
            logger=logger
        )
        return trainer
    elif predictor_type == 'SemiNAS':
        trainer = semi_nas_trainer(params=params,
                                   args=args,
                                   logger=logger)
        return trainer
    elif predictor_type == 'MLP':
        trainer = MetaNeuralnetTrainer(
            in_channel=params['in_channel'],
            num_layer=params['num_layers'],
            layer_width=params['layer_width'],
            lr=params['lr'],
            regularization=params['regularization'],
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            gpu=args.gpu,
            logger=logger
        )
        return trainer
    else:
        return TrainerGNN(gpu=args.gpu,
                          nas_benchmark=args.search_space,
                          predictor_type=predictor_type,
                          epoch_img_size=train_size,
                          load_model=load_model,
                          model_path=load_dir,
                          save_path=args.save_dir,
                          epochs=train_epochs,
                          with_g_func=args.with_g_func,
                          logger=logger,
                          **params)
