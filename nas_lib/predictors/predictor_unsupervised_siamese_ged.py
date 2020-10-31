# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from gnn_lib import GINConv, global_mean_pool as gmp


class PredictorSiameseGED(nn.Module):
    """
       without using share weights
       three gin layers
       feature concat
    """
    def __init__(self, input_dim=6, dim1=32, dim2=16):
        super(PredictorSiameseGED, self).__init__()
        layers = []

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

        # The first two GIN layer is shared by two pred
        nn1_residual = Sequential(Linear(input_dim, dim1, bias=True), ReLU(), Linear(dim1, dim1, bias=True))
        self.conv1_residual = GINConv(nn1_residual)
        self.bn1_residual = torch.nn.BatchNorm1d(dim1)

        nn2_residual = Sequential(Linear(dim1, dim1,  bias=True), ReLU(), Linear(dim1, dim1, bias=True))
        self.conv2_residual = GINConv(nn2_residual)
        self.bn2_residual = torch.nn.BatchNorm1d(dim1)

        nn3_residual = Sequential(Linear(dim1, dim1,  bias=True), ReLU(), Linear(dim1, dim1, bias=True))
        self.conv3_residual = GINConv(nn3_residual)
        self.bn3_residual = torch.nn.BatchNorm1d(dim1)

        self.linear_branch1 = torch.nn.Linear(dim1, dim1, bias=True)
        # branch1 head
        # self.linear_branch1_head = torch.nn.Linear(dim, dim, bias=True)
        # self.linear_branch1_out = torch.nn.Linear(dim, dim, bias=True)

        self.linear_branch2 = torch.nn.Linear(dim1, dim1, bias=True)
        # branch2 head
        # self.linear_branch2_head = torch.nn.Linear(dim, dim, bias=True)
        # self.linear_branch2_out = torch.nn.Linear(dim, dim, bias=True)

        layers.append(self.linear_branch1)
        layers.append(self.linear_branch2)

        # For residual predictor
        self.linear_before_residual = torch.nn.Linear(dim1*2, dim2, bias=True)
        self.linear_mean_residual = Linear(dim2, 1)

        self.output = torch.nn.Sigmoid()

        layers.append(self.linear_before_residual)
        layers.append(self.linear_mean_residual)

        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data_base, edge_index_base, batch_base, data_residual, edge_index_residual, batch_residual):
        return self.forward_batch(data_base, edge_index_base, batch_base, data_residual, edge_index_residual,
                                  batch_residual)

    def forward_batch(self, data_base, edge_index_base, batch_base, data_residual, edge_index_residual, batch_residual):
        # Base predictor inference
        x1_base = F.relu(self.conv1(data_base, edge_index_base))
        x1_base = self.bn1(x1_base)

        x2_base = F.relu(self.conv2_base(x1_base, edge_index_base))
        x2_base = self.bn2_base(x2_base)

        x3_base = F.relu(self.conv3_base(x2_base, edge_index_base))
        x3_base = self.bn3_base(x3_base)
        x_embedding_base = gmp(x3_base, batch_base)
        x_embedding_base = F.relu(self.linear_branch1(x_embedding_base))

        # Residual predictor inference
        x1_residual = F.relu(self.conv1_residual(data_residual, edge_index_residual))
        x1_residual = self.bn1_residual(x1_residual)

        x2_residual = F.relu(self.conv2_residual(x1_residual, edge_index_residual))
        x2_residual = self.bn2_residual(x2_residual)

        x3_residual = F.relu(self.conv3_residual(x2_residual, edge_index_residual))
        x3_residual = self.bn3_residual(x3_residual)
        x_embedding_residual = gmp(x3_residual, batch_residual)
        x_embedding_residual = F.relu(self.linear_branch2(x_embedding_residual))

        x_embedding_residual = torch.cat([x_embedding_base, x_embedding_residual], dim=-1)
        x_embedding_residual = F.relu(self.linear_before_residual(x_embedding_residual))
        outputs = self.linear_mean_residual(x_embedding_residual)

        outputs = self.output(outputs)

        return outputs
