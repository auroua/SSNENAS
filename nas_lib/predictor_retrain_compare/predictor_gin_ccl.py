# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from gnn_lib import GINConv, global_mean_pool as gmp


class PredictorGINCCL(nn.Module):
    """
       without using share weights
       three gin layers
       feature concat
    """
    def __init__(self, input_dim=8, dim1=32, num_classes=8, reTrain=False):
        # 64, 16
        super(PredictorGINCCL, self).__init__()
        layers = []
        # dim = 32
        # The first two GIN layer is shared by two pred
        nn1 = Sequential(Linear(input_dim, dim1, bias=True), ReLU(), Linear(dim1, dim1, bias=True))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim1)

        nn2_base = Sequential(Linear(dim1, dim1,  bias=True), ReLU(), Linear(dim1, dim1,  bias=True))
        self.conv2_base = GINConv(nn2_base)
        self.bn2_base = torch.nn.BatchNorm1d(dim1)

        nn3_base = Sequential(Linear(dim1, dim1,  bias=True), ReLU(), Linear(dim1, dim1,  bias=True))
        self.conv3_base = GINConv(nn3_base)
        self.bn3_base = torch.nn.BatchNorm1d(dim1)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(dim1, num_classes, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_classes, num_classes, bias=True)
        )

        self.fc.apply(self.init_weights)

        self.reTrain = reTrain
        if self.reTrain:
            self.output_layer = torch.nn.Sigmoid()

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.kaiming_uniform_(layer.weight, a=1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, data_base, edge_index_base, batch_base, t_sne=None):
        return self.forward_batch(data_base, edge_index_base, batch_base, t_sne=t_sne)

    def forward_batch(self, data_base, edge_index_base, batch_base, t_sne=False):
        # Base predictor inference
        x1_base = F.relu(self.conv1(data_base, edge_index_base))
        x1_base = self.bn1(x1_base)

        x2_base = F.relu(self.conv2_base(x1_base, edge_index_base))
        x2_base = self.bn2_base(x2_base)

        x3_base = F.relu(self.conv3_base(x2_base, edge_index_base))
        x3_base = self.bn3_base(x3_base)
        x_embedding_base = gmp(x3_base, batch_base)

        outputs = self.fc(x_embedding_base)

        if self.reTrain:
            outputs = self.output_layer(outputs)

        if t_sne:
            return outputs, x_embedding_base
        else:
            return outputs
