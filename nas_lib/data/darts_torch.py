import torch
from torch.utils.data import Dataset
import pickle
from nas_lib.data.collate_batch import BatchCollator


class DartsDataset(Dataset):
    def __init__(self, model_type, arch_path):
        super(DartsDataset, self).__init__()
        self.model_type = model_type
        with open(arch_path, 'rb') as fb:
            self.total_archs = pickle.load(fb)
        self.idxs = list(range(len(self.total_archs)))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        id = self.idxs[idx]
        path_encoding = self.total_archs[id]['pe_path_enc_aware_vec'][0]
        g_d = self.total_archs[id]['g_data']
        return g_d, path_encoding

    def __str__(self):
        return f'This dataset contains {len(self.total_archs)} architectures.'


if __name__ == '__main__':
    nasbench_201_dataset = DartsDataset('SS_CCL',
                                        '/home/aurora/data_disk_new/dataset_train/nas_bench_301/gen_archs/data_info_part1.pkl')
    print(nasbench_201_dataset)
    device = torch.device('cuda:0')
    dataset_loader = torch.utils.data.DataLoader(nasbench_201_dataset,
                                                 batch_size=2000, shuffle=True,
                                                 num_workers=4, collate_fn=BatchCollator())
    for sample_data in dataset_loader:
        g_d, path_encodings = sample_data
        # archs = g_d.to(device)
        path_encodings = path_encodings.to(device)
        print(len(g_d), len(path_encodings))