import torch
from torch.utils.data import Dataset
from nas_lib.data.nasbench_201 import NASBench201
import numpy as np
from nas_lib.utils.utils_data import NASBENCH_201_OPS, nasbench2graph_201
from gnn_lib.data import Data


class NASBenche201Dataset(Dataset):
    def __init__(self, model_type):
        super(NASBenche201Dataset, self).__init__()
        self.total_keys, self.total_archs = self._load_data()
        self.idxs = list(range(len(self.total_keys)))
        self.model_type = model_type

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        id = self.idxs[idx]

        if self.model_type == 'moco':
            arch = self.total_archs[self.total_keys[id]][0][0]
            ops = self.total_archs[self.total_keys[id]][0][1]
            path_encoding = self.total_archs[self.total_keys[id]][-1]
            return arch, ops, path_encoding
        elif self.model_type == 'SS_CCL':
            arch = self.total_archs[self.total_keys[id]][0][0]
            ops = self.total_archs[self.total_keys[id]][0][1]
            path_encoding = self.total_archs[self.total_keys[id]][-1]

            edge_index, node_f = nasbench2graph_201((arch, ops), is_idx=True)
            g_d = Data(edge_index=edge_index.long(), x=node_f.float())
            return g_d, path_encoding
        else:
            raise NotImplementedError(f'The model type {self.model_type} does not support!')

    def __str__(self):
        return f'This dataset contains {len(self.total_keys)} architectures.'

    def _load_data(self):
        nasbench_201 = NASBench201()
        total_keys = nasbench_201.total_keys
        total_archs = nasbench_201.total_archs
        for k, v in total_archs.items():
            total_archs[k][0] = [v[0][0], np.array([NASBENCH_201_OPS.index(op) for op in v[0][1]])]
        del nasbench_201
        return total_keys, total_archs


if __name__ == '__main__':
    nasbench_201_dataset = NASBenche201Dataset()
    print(nasbench_201_dataset)
    device = torch.device('cuda:0')
    dataset_loader = torch.utils.data.DataLoader(nasbench_201_dataset,
                                                 batch_size=2000, shuffle=True,
                                                 num_workers=4)
    for sample_data in dataset_loader:
        archs, ops, path_encodings = sample_data
        archs = archs.to(device)
        ops = ops.to(device)
        path_encodings = path_encodings.to(device)
        print(len(archs), len(ops), len(path_encodings))