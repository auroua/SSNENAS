from .agent import BaseAgent
from gnn_lib import GCNConv, global_mean_pool as gmp
import torch.nn.functional as F
import torch
import torch.nn as nn

# writer = SummaryWriter('./gin')


class NasBenchGCNNnpCasAgent(BaseAgent):
    def __init__(self, input_dim):
        super(NasBenchGCNNnpCasAgent, self).__init__()
        dim = 144
        dim2 = 128
        self.conv1 = GCNConv(input_dim, dim, improved=False, cached=False)
        self.conv2 = GCNConv(dim, dim, improved=False, cached=False)
        self.conv3 = GCNConv(dim, dim, improved=False, cached=False)

        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.conv1_reverse = GCNConv(input_dim, dim, improved=False, cached=False)
        self.conv2_reverse = GCNConv(dim, dim, improved=False, cached=False)
        self.conv3_reverse = GCNConv(dim, dim, improved=False, cached=False)

        self.linear = torch.nn.Linear(dim, dim2)
        self.liner2 = torch.nn.Linear(dim2, 2)

        self.out_layer = torch.nn.Sigmoid()
        layers = []
        layers.append(self.linear)
        layers.append(self.liner2)
        layers.append(self.out_layer)

        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data, edge_index, batch, data_reverse, edge_index_reverse):
        return self.forward_batch(data, edge_index, batch, data_reverse, edge_index_reverse)

    def forward_batch(self, data, edge_index, batch, data_reverse, edge_index_reverse):
        x1 = (F.relu(self.conv1(data, edge_index)) + F.relu(self.conv1_reverse(data_reverse, edge_index_reverse)))*0.5
        x1 = self.bn1(x1)
        x1 = F.dropout(x1, training=self.training, p=0.1)
        x2 = (F.relu(self.conv2(x1, edge_index)) + F.relu(self.conv2_reverse(x1, edge_index_reverse)))*0.5
        x2 = self.bn2(x2)
        x2 = F.dropout(x2, training=self.training, p=0.1)
        x3 = (F.relu(self.conv3(x2, edge_index)) + F.relu(self.conv3_reverse(x2, edge_index_reverse)))*0.5
        x3 = self.bn3(x3)
        x3 = F.dropout(x3, training=self.training, p=0.1)

        x_embedding = gmp(x3, batch)
        x_embedding = F.relu(self.linear(x_embedding))
        x_embedding = F.dropout(x_embedding, p=0.1, training=self.training)
        output = self.liner2(x_embedding)
        output = self.out_layer(output)
        return output