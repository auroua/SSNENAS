import torch
from torch.utils.data import Dataset
import numpy as np
from nas_lib.utils.utils_data import NASBENCH_101_OPS, nasbench2graph_101
import pickle
from configs import nas_bench_101_converted_file_path
from gnn_lib.data import Data


class NASBenche101Dataset(Dataset):
    def __init__(self, model_type):
        super(NASBenche101Dataset, self).__init__()
        self.total_keys, self.total_archs = self._load_data()
        self.idxs = list(range(len(self.total_keys)))
        self.model_type = model_type

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        id = self.idxs[idx]

        if self.model_type == 'moco':
            arch = self.total_archs[self.total_keys[id]]['matrix']
            ops = self.total_archs[self.total_keys[id]]['ops']
            path_encoding = self.total_archs[self.total_keys[id]]['path_indices']
            return arch, ops, path_encoding
        elif self.model_type == 'SS_CCL':
            arch = self.total_archs[self.total_keys[id]]['matrix']
            ops = self.total_archs[self.total_keys[id]]['ops']
            path_encoding = self.total_archs[self.total_keys[id]]['path_indices']

            edge_index, node_f = nasbench2graph_101((arch, ops), is_idx=True)
            g_d = Data(edge_index=edge_index.long(), x=node_f.float())
            return g_d, path_encoding
        else:
            raise NotImplementedError(f'Model type {self.model_type} does not support at present!')

    def __str__(self):
        return f'This dataset contains {len(self.total_keys)} architectures.'

    def _load_data(self):
        with open(nas_bench_101_converted_file_path, 'rb') as fb:
            total_keys = pickle.load(fb)
            total_archs = pickle.load(fb)
        for k, v in total_archs.items():
            total_archs[k]['ops'] = np.array([NASBENCH_101_OPS[op] for op in v['ops']], dtype=np.int16)
        return total_keys, total_archs


if __name__ == '__main__':
    nasbench_101_dataset = NASBenche101Dataset()
    print(nasbench_101_dataset)
    device = torch.device('cuda:0')
    dataset_loader = torch.utils.data.DataLoader(nasbench_101_dataset,
                                                 batch_size=2000, shuffle=True,
                                                 num_workers=4)
    for sample_data in dataset_loader:
        archs, ops, path_encodings = sample_data
        print(len(archs), len(ops), len(path_encodings))
        archs = archs.to(device)
        ops = ops.to(device)
        path_encodings = path_encodings.to(device)
