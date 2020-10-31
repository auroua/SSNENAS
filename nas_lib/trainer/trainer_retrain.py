# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved


from nas_lib.params import get_params
from nas_lib.trainer.trainer_gnn import TrainerGNN


class NASBenchReTrain:
    def __init__(self, args, predictor_type, train_size, load_model=False, load_dir=None, train_epochs=None,
                 logger=None):
        params = get_params(args, predictor_type)
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
        # test_matrix = [(d['matrix'], d['ops']) for d in test_data]
        ytest = [d['val_acc'] for d in test_data]
        g_test_data = [d['g_data'] for d in test_data]
        spearman_corr, kendalltau_corr = self.trainer.fit_g(ytrain, g_train_data, ytest, g_test_data)
        return spearman_corr, kendalltau_corr


def build_predictor(predictor_type, train_size, params, args, load_model, load_dir, train_epochs, logger):
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