import os
from nas_lib.data.darts_api.arch_darts import ArchDarts
import pickle
from tools_darts.gen_darts_archs import convert_genotype_form, OPS
from nas_lib.data.darts import DataSetDarts
from nas_lib.utils.utils_darts import nasbench2graph2
from gnn_lib.data import Data
from tools_darts.gen_darts_archs import Genotype
import pickle


SEQ_LEN = 612


def gen_random_darts_models(save_path):
    ArchDarts(None).generate_normal_archs(save_path=save_path)


def gen_darts_dataset(base_path, save_path=None):
    files = [os.path.join(base_path, f) for f in os.listdir(base_path) if not f.endswith('txt')]
    all_archs = []
    darts_dataset = DataSetDarts()
    for f in files:
        with open(f, 'rb') as fb:
            genotype_list = pickle.load(fb)
            keys_list = pickle.load(fb)
        for (genotype, key) in zip(genotype_list, keys_list):
            if len(all_archs) % 10000 == 0 and len(all_archs) != 0:
                print(f'{len(all_archs)} have processed!')
            f_new = convert_genotype_form(genotype, OPS)
            arch = (f_new.normal, f_new.reduce)
            arch_darts = ArchDarts(arch)
            path_encoding_position_aware = arch_darts.get_path(
                path_type='path_enc_aware_vec',
                seq_len=612
            )
            path_encoding = arch_darts.get_path(
                path_type='path_enc_vec',
                seq_len=612
            )
            path_adj_encoding = arch_darts.get_path(
                path_type='adj_enc_vec',
                seq_len=612
            )

            matrix, ops = darts_dataset.assemble_graph_from_single_arch(arch)
            edge_indices, node_features = nasbench2graph2((matrix, ops))
            edge_reverse_indices, node_reverse_features = nasbench2graph2((matrix, ops), reverse=True)

            all_archs.append(
                {
                    'matrix': matrix,
                    'ops': ops,
                    'pe_adj_enc_vec': path_adj_encoding,
                    'pe_path_enc_vec': path_encoding,
                    'pe_path_enc_aware_vec': path_encoding_position_aware,
                    'hash_key': key,
                    'genotype': genotype,
                    'edge_idx': edge_indices,
                    'node_f': node_features,
                    'g_data': Data(edge_index=edge_indices.long(), x=node_features.float()),
                    'edge_idx_reverse': edge_reverse_indices,
                    'node_f_reverse': node_reverse_features,
                    'g_data_reverse': Data(edge_index=edge_reverse_indices.long(), x=node_reverse_features.float()),
                }
            )
    if save_path:
        with open(save_path, 'wb') as fb:
            pickle.dump(all_archs, fb)
    return all_archs


if __name__ == '__main__':
    save_path = '/home/aurora/data_disk_new/train_output_2021/darts_save_path/architectures'
    gen_darts_dataset(save_path, save_path='/home/aurora/data_disk_new/train_output_2021/darts_save_path/architectures/part3_partial.pkl')