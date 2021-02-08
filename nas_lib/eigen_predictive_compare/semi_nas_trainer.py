from torch.nn import Module
import torch
import time
import pickle
from nas_lib.predictors_compare.SemiNAS.controller import NAO
import numpy as np
from nas_lib.predictors_compare.SemiNAS import utils
import torch.nn.functional as F
from nas_lib.utils.comm import get_spearmanr_coorlection, get_kendalltau_coorlection


class semi_nas_trainer(Module):
    def __init__(self, params, args, logger):
        super(semi_nas_trainer, self).__init__()
        self.predictor = NAO(
            params['encoder_layers'],
            params['decoder_layers'],
            params['mlp_layers'],
            params['hidden_size'],
            params['mlp_hidden_size'],
            params['vocab_size'],
            params['dropout'],
            params['source_length'],
            params['encoder_length'],
            params['decoder_length'],
        )
        self.params = params
        self.logger = logger
        self.device = torch.device(f'cuda:{args.gpu}')
        self.predictor.to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=params['lr'], weight_decay=params['l2_reg'])

    def forward(self):
        pass

    def fit_g(self, ytrain, train_archs, ytest, test_archs):
        # min_val = min(ytrain)
        # max_val = max(ytrain)
        # train_target = [(i - min_val) / (max_val - min_val) for i in ytrain]
        # ytest_target = [(j - min_val) / (max_val - min_val) for j in ytest]
        train_target = ytrain
        ytest_target = ytest
        controller_train_dataset = utils.ControllerDataset(train_archs, train_target, True)
        controller_train_queue = torch.utils.data.DataLoader(
            controller_train_dataset, batch_size=self.params['batch_size'], shuffle=True, pin_memory=True)
        start = time.time()
        for epoch in range(1, self.params['epochs'] + 1):
            loss, mse, ce = self.controller_train(controller_train_queue, self.predictor, self.optimizer)
            # self.logger.info("epoch %04d train loss %.6f mse %.6f ce %.6f", epoch, loss, mse, ce)
        duration = time.time() - start
        test_seq = [test_vec for test_vec in test_archs]
        controller_synthetic_dataset = utils.ControllerDataset(test_seq, None, False)
        controller_synthetic_queue = torch.utils.data.DataLoader(controller_synthetic_dataset,
                                                                 batch_size=64, shuffle=False, pin_memory=True)
        pred_test_target = []
        with torch.no_grad():
            self.predictor.eval()
            for sample in controller_synthetic_queue:
                encoder_input = sample['encoder_input'].to(self.device)
                _, _, _, predict_value = self.predictor.encoder(encoder_input)
                pred_test_target += predict_value.data.squeeze().tolist()
        s_t = get_spearmanr_coorlection(pred_test_target, ytest_target)[0]
        k_t = get_kendalltau_coorlection(pred_test_target, ytest_target)[0]
        if self.logger:
            self.logger.info(f'Training time cost: {duration}, Spearman Correlation: {s_t}, '
                             f'Kendalltau Corrlation: {k_t}')
        else:
            print(f'Training time cost: {duration}, '
                  f'Spearman Correlation: {s_t}, '
                  f'Kendalltau Corrlation: {k_t}')
        return s_t, k_t, duration

    def controller_train(self, train_queue, model, optimizer):
        objs = utils.AvgrageMeter()
        mse = utils.AvgrageMeter()
        nll = utils.AvgrageMeter()
        model.train()
        for step, sample in enumerate(train_queue):
            encoder_input = utils.move_to_cuda(sample['encoder_input'], self.device)
            encoder_target = utils.move_to_cuda(sample['encoder_target'], self.device)
            decoder_input = utils.move_to_cuda(sample['decoder_input'], self.device)
            decoder_target = utils.move_to_cuda(sample['decoder_target'], self.device)

            optimizer.zero_grad()
            predict_value, log_prob, arch = model(encoder_input, decoder_input)
            loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze())
            loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1))
            loss = self.params['trade_off'] * loss_1 + (1 - self.params['trade_off']) * loss_2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.params['grad_bound'])
            optimizer.step()

            n = encoder_input.size(0)
            objs.update(loss.data, n)
            mse.update(loss_1.data, n)
            nll.update(loss_2.data, n)

        return objs.avg, mse.avg, nll.avg