import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from nas_lib.utils.utils_data import gen_batch_idx
from nas_lib.utils.comm import get_spearmanr_coorlection, get_kendalltau_coorlection
import time


class MetaNeuralnetTorch(nn.Module):
    def __init__(self, in_channel, num_layer, layer_width):
        super(MetaNeuralnetTorch, self).__init__()
        total_layers = []
        self.input_layer = nn.Linear(in_channel, layer_width[0])
        total_layers.append(self.input_layer)
        self.model_keys = []
        # for i in range(1, num_layer):
        for idx, layer_w in enumerate(layer_width):
            if idx == len(layer_width) - 1:
                break
            self.add_module('layer_%d' % (idx+1), nn.Linear(layer_width[idx], layer_width[idx+1]))
            self.model_keys.append('layer_%d' % (idx+1))
            total_layers.append(getattr(self, 'layer_%d' % (idx+1)))
        self.output_layer = nn.Linear(layer_width[-1], 1)
        total_layers.append(self.output_layer)
        for layer in total_layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for k in self.model_keys:
            x = F.relu(getattr(self, k)(x))
        out = self.output_layer(x)
        return out


class MetaNeuralnetTrainer:
    def __init__(self, in_channel, num_layer, layer_width, lr, regularization, epochs, batch_size, gpu, logger=None):
        self.net = MetaNeuralnetTorch(in_channel, num_layer, layer_width)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=regularization)
        self.epochs = epochs
        self.batch_size = batch_size
        self.logger = logger
        self.device = torch.device('cuda: %d' % gpu)
        self.net.to(self.device)

    def train(self, train_data, target_data):
        self.net.train()
        for _ in range(self.epochs):
            idx_list = list(range(train_data.shape[0]))
            random.shuffle(idx_list)
            batch_idx_list = gen_batch_idx(idx_list, self.batch_size)
            for batch in batch_idx_list:
                data_list = []
                target_list = []
                for idx in batch:
                    data_list.append(train_data[idx])
                    target_list.append(target_data[idx])
                train_d = torch.tensor(data_list, dtype=torch.float32)
                target_d = torch.tensor(target_list, dtype=torch.float32)
                train_d = train_d.to(self.device)
                target_d = target_d.to(self.device)
                out = self.net(train_d)
                out = torch.squeeze(out, dim=1)
                loss = torch.mean(torch.abs(out - target_d))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        error = self.train_error(train_data, target_data)
        if self.logger:
            self.logger.info('Meta neural network training loss is %.5f' % error.item())
        return error

    def pred(self, val_data):
        self.net.eval()
        pred_list = []
        idx_list = list(range(len(val_data)))
        batch_idx_list = gen_batch_idx(idx_list, 32)
        with torch.no_grad():
            for batch_idx in batch_idx_list:
                data_list = []
                for idx in batch_idx:
                    data_list.append(val_data[idx])
                val_d = torch.tensor(data_list, dtype=torch.float32)
                val_d = val_d.to(self.device)
                out = self.net(val_d)
                out = torch.squeeze(out, dim=1)
                pred_list.append(out)
        return torch.cat(pred_list, dim=0)

    def train_error(self, train_data, target_data):
        output = self.pred(train_data).cpu().numpy()
        target_data = np.array(target_data)
        error = np.mean(np.abs(output-target_data))
        return error

    def fit_g(self, ytrain, train_data, ytest, test_data):
        ytrain = np.array(ytrain)
        train_data = np.array(train_data)
        ytest = np.array(ytest)
        test_data = np.array(test_data)

        start = time.time()
        self.train(train_data, ytrain)
        duration = time.time() - start
        pred = self.pred(test_data)
        pred = pred.detach().cpu().numpy()
        s_t = get_spearmanr_coorlection(pred, ytest)[0]
        k_t = get_kendalltau_coorlection(pred, ytest)[0]
        if self.logger:
            self.logger.info(f'Training time cost: {duration}, Spearman Correlation: {s_t}, '
                             f'Kendalltau Corrlation: {k_t}')
        else:
            print(f'Training time cost: {duration}, '
                  f'Spearman Correlation: {s_t}, '
                  f'Kendalltau Corrlation: {k_t}')
        return s_t, k_t, duration


if __name__ == '__main__':
    trainer = MetaNeuralnetTrainer(364, 3, 20, 0.1, 0, 200, 32, 0)
    train_data = np.random.randn(1000, 364)
    target_data = np.random.randn(1000)

    trainer.train(train_data, target_data)

    print(trainer.train_error(train_data, target_data))