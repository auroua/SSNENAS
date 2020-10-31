# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved


from nas_lib.params import get_params
from .trainer_gnn_embedding import TrainerGED


class NASBenchTrainer:
    def __init__(self, args, predictor_type, train_size, train_epochs, logger=None):
        params = get_params(args, predictor_type)
        if args.save_model:
            params['save_model'] = True
        self.trainer = build_trainer(predictor_type=predictor_type,
                                     params=params,
                                     args=args,
                                     train_size=train_size,
                                     train_epochs=train_epochs,
                                     logger=logger)
        self.predictor_type = predictor_type
        self.params = params

    def fit_unsupervised(self, all_data):
        train_archs = [(d['matrix'], d['ops'], d['pe_path_enc_aware_vec']) for d in all_data]
        self.trainer.fit(train_archs)


def build_trainer(predictor_type, train_size, params, args, train_epochs, logger=None):
    if predictor_type == 'SS_RL':
        return TrainerGED(gpu=args.gpu,
                          nas_benchmark=args.search_space,
                          predictor_type=predictor_type,
                          epoch_img_size=train_size,
                          model_save_dir=args.save_dir,
                          ratio=args.train_ratio,
                          epochs=train_epochs,
                          logger=logger,
                          **params)
    else:
        raise NotImplementedError()