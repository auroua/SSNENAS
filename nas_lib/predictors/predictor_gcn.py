# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved

from gnn_lib import GCNConv, global_mean_pool as gmp, global_max_pool as gap
import torch.nn.functional as F
import torch
import torch.nn as nn


class PredictorGCN(nn.Module):
    def __init__(self, input_dim, dim=16):
        super(PredictorGCN, self).__init__()
        layers = []

        self.conv1 = GCNConv(input_dim, dim, cached=False)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.conv2 = GCNConv(dim, dim, cached=False)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.linear_before = torch.nn.Linear(dim*2, 16)
        self.linear_mean = torch.nn.Linear(16, 1)

        self.out_layer = torch.nn.Sigmoid()

        layers.append(self.conv1)
        layers.append(self.conv2)
        layers.append(self.linear_before)
        layers.append(self.linear_mean)

        for layer in layers:
            if isinstance(layer, GCNConv):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data, edge_index, batch):
        return self.forward_batch(data, edge_index, batch)

    def forward_batch(self, data, edge_index, batch):
        x1 = F.relu(self.conv1(data, edge_index))
        x1 = self.bn1(x1)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = self.bn2(x2)

        x_embedding = torch.cat([gmp(x2, batch), gap(x2, batch)], dim=1)
        x_embedding = F.relu(self.linear_before(x_embedding))
        x_embedding = F.dropout(x_embedding, p=0.1, training=self.training)

        pred = self.linear_mean(x_embedding)
        pred = self.out_layer(pred)

        return pred