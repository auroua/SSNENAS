# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from gnn_lib import GINConv, global_mean_pool as gmp


class PredictorGIN(nn.Module):
    def __init__(self, input_dim=6, dim1=32, dim2=16, ):
        super(PredictorGIN, self).__init__()
        layers = []
        nn1 = Sequential(Linear(input_dim, dim1, bias=True), ReLU(), Linear(dim1, dim1, bias=True))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim1)

        nn2 = Sequential(Linear(dim1, dim1,  bias=True), ReLU(), Linear(dim1, dim1,  bias=True))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim1)
        #
        nn3 = Sequential(Linear(dim1, dim1), ReLU(), Linear(dim1, dim1))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim1)

        self.linear_before = torch.nn.Linear(dim1, dim2, bias=True)

        self.linear_mean = Linear(dim2, 1)
        layers.append(self.linear_mean)
        layers.append(self.linear_before)
        self.out_layer = torch.nn.Sigmoid()

        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data, edge_index, batch, t_sne=None):
        return self.forward_batch(data, edge_index, batch, t_sne=t_sne)

    def forward_batch(self, data, edge_index, batch, t_sne=None):
        x1 = F.relu(self.conv1(data, edge_index))
        x1 = self.bn1(x1)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = self.bn2(x2)

        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = self.bn3(x3)

        x_embedding = gmp(x3, batch)
        x_embedding_mean = F.relu(self.linear_before(x_embedding))
        x_embedding_drop = F.dropout(x_embedding_mean, p=0.1, training=self.training)

        pred = self.linear_mean(x_embedding_drop)
        pred = self.out_layer(pred)

        if t_sne:
            return pred, x_embedding
        else:
            return pred