import numpy as np
from hashlib import sha256
import random
from collections import defaultdict
from nas_lib.algos import algo_sort
import pickle
import sys
import os


OPS = ['none',
       'max_pool_3x3',
       'avg_pool_3x3',
       'skip_connect',
       'sep_conv_3x3',
       'sep_conv_5x5',
       'dil_conv_3x3',
       'dil_conv_5x5'
       ]

OPS_mapping = {'none': 0,
               'max_pool_3x3': 1,
               'avg_pool_3x3': 2,
               'skip_connect': 3,
               'sep_conv_3x3': 4,
               'sep_conv_5x5': 5,
               'dil_conv_3x3': 6,
               'dil_conv_5x5': 7
               }
NUM_VERTICES = 4
INPUT_1 = 'c_k-2'
INPUT_2 = 'c_k-1'


class ArchDarts:
    def __init__(self, arch):
        self.arch = arch

    @classmethod
    def random_arch(cls):
        # output a uniformly random architecture spec
        # from the DARTS repository
        # https://github.com/quark0/darts

        normal = []
        reduction = []
        for i in range(NUM_VERTICES):
            ops = np.random.choice(range(len(OPS)), NUM_VERTICES)

            # input nodes for conv
            nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
            # input nodes for reduce
            nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)

            normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
            reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])
        return (normal, reduction)

    def mutate(self, edits):
        """ mutate a single arch """
        # first convert tuple to array so that it is mutable
        mutation = []
        for cell in self.arch:
            mutation.append([])
            for pair in cell:
                mutation[-1].append([])
                for num in pair:
                    mutation[-1][-1].append(num)
        # make mutations
        for _ in range(edits):
            cell = np.random.choice(2)
            pair = np.random.choice(len(OPS))
            num = np.random.choice(2)
            if num == 1:
                mutation[cell][pair][num] = np.random.choice(len(OPS))
            else:
                inputs = pair // 2 + 2
                choice = np.random.choice(inputs)
                if pair % 2 == 0 and mutation[cell][pair + 1][num] != choice:
                    mutation[cell][pair][num] = choice
                elif pair % 2 != 0 and mutation[cell][pair - 1][num] != choice:
                    mutation[cell][pair][num] = choice
        return mutation

    def get_paths(self):
        """ return all paths from input to output """

        path_builder = [[[], [], [], []], [[], [], [], []]]
        paths = [[], []]

        for i, cell in enumerate(self.arch):
            for j in range(len(OPS)):
                if cell[j][0] == 0:
                    path = [INPUT_1, OPS[cell[j][1]]]
                    path_builder[i][j//2].append(path)
                    paths[i].append(path)
                elif cell[j][0] == 1:
                    path = [INPUT_2, OPS[cell[j][1]]]
                    path_builder[i][j//2].append(path)
                    paths[i].append(path)
                else:
                    for path in path_builder[i][cell[j][0] - 2]:
                        path = [*path, OPS[cell[j][1]]]
                        path_builder[i][j//2].append(path)
                        paths[i].append(path)

        return paths

    def get_path_indices(self, long_paths=True):
        """
        compute the index of each path
        There are 4 * (8^0 + ... + 8^4) paths total
        If long_paths = False, we give a single boolean to all paths of
        size 4, so there are only 4 * (1 + 8^0 + ... + 8^3) paths
        """
        paths = self.get_paths()
        normal_paths, reduce_paths = paths
        num_ops = len(OPS)
        """
        Compute the max number of paths per input per cell.
        Since there are two cells and two inputs per cell, 
        total paths = 4 * max_paths
        """

        max_paths = sum([num_ops ** i for i in range(NUM_VERTICES + 1)])
        path_indices = []

        # set the base index based on the cell and the input
        for i, paths in enumerate((normal_paths, reduce_paths)):
            for path in paths:
                index = i * 2 * max_paths
                if path[0] == INPUT_2:
                    index += max_paths

                # recursively compute the index of the path
                for j in range(NUM_VERTICES + 1):
                    if j == len(path) - 1:
                        path_indices.append(index)
                        break
                    elif j == (NUM_VERTICES - 1) and not long_paths:
                        path_indices.append(2 * (i + 1) * max_paths - 1)
                        break
                    else:
                        index += num_ops ** j * (OPS.index(path[j + 1]) + 1)
        return tuple(path_indices)

    def encode_paths(self, cutoff=None):
        # output one-hot encoding of paths
        path_indices = self.get_path_indices()
        num_ops = len(OPS)

        max_paths = sum([num_ops ** i for i in range(NUM_VERTICES + 1)])

        path_encoding = np.zeros(4 * max_paths)
        for index in path_indices:
            path_encoding[index] = 1
        if cutoff:
            path_encoding = path_encoding[:cutoff]
        return path_encoding

    def get_position_aware_paths(self):
        """ return all paths from input to output """

        path_builder_normal = [[], [], [], []]
        path_builder_idxs_normal = [[], [], [], []]
        paths_normal = []
        paths_idxs_normal = []
        normal_cell = self.arch[0]
        reduction_cell = self.arch[1]

        for j in range(len(OPS)):
            if normal_cell[j][0] == 0:
                path = [INPUT_1, OPS[normal_cell[j][1]]]
                path_builder_normal[j//2].append(path)
                path_builder_idxs_normal[j//2].append([0, j//2+2])
                paths_normal.append(path)
                paths_idxs_normal.append([0, j//2+2])
            elif normal_cell[j][0] == 1:
                path = [INPUT_2, OPS[normal_cell[j][1]]]
                path_builder_normal[j//2].append(path)
                path_builder_idxs_normal[j//2].append([1, j//2+2])
                paths_normal.append(path)
                paths_idxs_normal.append([1, j//2+2])
            else:
                for idx_p, path in enumerate(path_builder_normal[normal_cell[j][0] - 2]):
                    path = [*path, OPS[normal_cell[j][1]]]
                    path_idxs = [*path_builder_idxs_normal[normal_cell[j][0] - 2][idx_p], j//2+2]
                    path_builder_normal[j//2].append(path)
                    path_builder_idxs_normal[j//2].append(path_idxs)
                    paths_normal.append(path)
                    paths_idxs_normal.append(path_idxs)

        paths_builder_reduction = [[], [], [], []]
        path_builder_idxs_reduction = [[], [], [], []]
        paths_reduction = []
        paths_idxs_reduction = []

        for j in range(len(OPS)):
            if reduction_cell[j][0] == 0 or reduction_cell[j][0] == 1:
                for idx_r_p, path in enumerate(paths_normal):
                    path = [*path, OPS[reduction_cell[j][1]]]
                    path_idxs = [*paths_idxs_normal[idx_r_p], j//2+6]

                    paths_builder_reduction[j//2].append(path)
                    path_builder_idxs_reduction[j//2].append(path_idxs)
                    paths_reduction.append(path)
                    paths_idxs_reduction.append(path_idxs)
            else:
                for idx_p, path in enumerate(paths_builder_reduction[reduction_cell[j][0] - 2]):
                    path = [*path, OPS[reduction_cell[j][1]]]
                    path_idxs = [*path_builder_idxs_reduction[reduction_cell[j][0] - 2][idx_p], j//2+6]

                    paths_builder_reduction[j//2].append(path)
                    path_builder_idxs_reduction[j//2].append(path_idxs)
                    paths_reduction.append(path)
                    paths_idxs_reduction.append(path_idxs)
        return paths_reduction, paths_idxs_reduction

    def get_position_aware_paths_sep(self, cell):
        """ return all paths from input to output """
        path_builder = [[], [], [], []]
        path_builder_idxs = [[], [], [], []]
        paths = []
        paths_idxs = []
        for j in range(len(OPS)):
            if cell[j][0] == 0:
                path = [INPUT_1, OPS[cell[j][1]]]
                path_builder[j//2].append(path)
                path_builder_idxs[j//2].append([0, j//2+2])
                paths.append(path)
                paths_idxs.append([0, j//2+2])
            elif cell[j][0] == 1:
                path = [INPUT_2, OPS[cell[j][1]]]
                path_builder[j//2].append(path)
                path_builder_idxs[j//2].append([1, j//2+2])
                paths.append(path)
                paths_idxs.append([1, j//2+2])
            else:
                for idx_p, path in enumerate(path_builder[cell[j][0] - 2]):
                    path = [*path, OPS[cell[j][1]]]
                    path_idxs = [*path_builder_idxs[cell[j][0] - 2][idx_p], j//2+2]
                    path_builder[j//2].append(path)
                    path_builder_idxs[j//2].append(path_idxs)
                    paths.append(path)
                    paths_idxs.append(path_idxs)
        return paths, paths_idxs

    def sort_seqs_list(self, paths, paths_idx):
        seq_len_dict = defaultdict(list)
        for idx, p in enumerate(paths):
            seq_len_dict[len(p)].append(idx)
        k_sorted = sorted(list(seq_len_dict.keys()))
        sorted_idxs = []
        for k in k_sorted:
            paths_v = [(v_i, paths_idx[v_i]) for v_i in seq_len_dict[k]]
            sort_results = algo_sort.quick_sort_list(paths_v)
            sorted_idxs.extend([k[0] for k in sort_results])
        return [paths[idx] for idx in sorted_idxs], [paths_idx[idx] for idx in sorted_idxs]

    def get_position_aware_path_indices(self, cell=None, seq_len=1224):
        """
        compute the index of each path
        There are 4 * (8^0 + ... + 8^4) paths total
        If long_paths = False, we give a single boolean to all paths of
        size 4, so there are only 4 * (1 + 8^0 + ... + 8^3) paths
        """
        paths, paths_idxs = self.get_position_aware_paths_sep(cell)
        path, path_idxs = self.sort_seqs_list(paths, paths_idxs)

        vectors_list = []
        for (p_list, idx_list) in zip(paths, path_idxs):
            vec = np.zeros(2+NUM_VERTICES*len(OPS), dtype=np.int16)
            for p, ids in zip(p_list, idx_list):
                if ids < 2:
                    vec[ids] = 1
                else:
                    vec[2 + (ids-2)*len(OPS) + OPS_mapping[p]] = 1
            vectors_list.append(vec)
        path_encoding = np.array(vectors_list, dtype=np.int16)
        path_encoding = path_encoding.reshape((1, -1))[0]
        residual_len = seq_len - path_encoding.shape[0]
        if residual_len != 0:
            residual_np = np.zeros(residual_len, dtype=np.int16)
            path_encoding = np.concatenate([path_encoding, residual_np])
        return path_encoding

    def encode_position_aware_paths(self, seq_len, arch=None):
        # output one-hot encoding of paths
        if arch:
            normal_cell, reduction_cell = arch
        else:
            normal_cell, reduction_cell = self.arch
        path_encoding_normal = self.get_position_aware_path_indices(normal_cell, seq_len=seq_len)
        path_encoding_reduction = self.get_position_aware_path_indices(reduction_cell, seq_len=seq_len)

        path_encoding = np.concatenate([path_encoding_normal, path_encoding_reduction])
        return path_encoding, (path_encoding_normal, path_encoding_reduction)

    def encode_adj(self):
        matrices = []
        ops = []
        true_num_vertices = NUM_VERTICES + 3
        for cell in self.arch:
            matrix = np.zeros((true_num_vertices, true_num_vertices))
            op_list = []
            for i, edge in enumerate(cell):
                dest = i//2 + 2
                matrix[edge[0]][dest] = 1
                op_list.append(edge[1])
            for i in range(2, 6):
                matrix[i][-1] = 1
            matrices.append(matrix)
            ops.append(op_list)

        encoding = [*matrices[0].flatten(), *ops[0], *matrices[1].flatten(), *ops[1]]
        return np.array(encoding)

    def get_neighborhood(self,
                         nasbench,
                         mutate_encoding='adj',
                         cutoff=None,
                         index_hash=None,
                         shuffle=True):
        if mutate_encoding != 'adj':
            print('{} is not yet implemented as a neighborhood for nasbench301'.format(mutate_encoding))
            sys.exit()

        op_nbhd = []
        edge_nbhd = []

        for i, cell in enumerate(self.arch):
            for j, pair in enumerate(cell):

                # mutate the op
                available = [op for op in range(len(OPS)) if op != pair[1]]
                for op in available:
                    new_arch = self.make_mutable()
                    new_arch[i][j][1] = op
                    op_nbhd.append({'arch': new_arch})

                # mutate the edge
                other = j + 1 - 2 * (j % 2)
                available = [edge for edge in range(j//2+2) \
                             if edge not in [cell[other][0], pair[0]]]

                for edge in available:
                    new_arch = self.make_mutable()
                    new_arch[i][j][0] = edge
                    edge_nbhd.append({'arch': new_arch})

        if shuffle:
            random.shuffle(edge_nbhd)
            random.shuffle(op_nbhd)

        # 112 in edge nbhd, 24 in op nbhd
        # alternate one edge nbr per 4 op nbrs
        nbrs = []
        op_idx = 0
        for i in range(len(edge_nbhd)):
            nbrs.append(edge_nbhd[i])
            for j in range(4):
                nbrs.append(op_nbhd[op_idx])
                op_idx += 1
        nbrs = [*nbrs, *op_nbhd[op_idx:]]

        return nbrs

    def path_distance(self, other):
        # compute the distance between two architectures
        # by comparing their path encodings
        return np.sum(np.array(self.encode_paths() != np.array(other.encode_paths())))

    def get_path(self, path_type, seq_len, cut_off=None):
        if path_type == 'adj_enc_vec':
            return self.encode_adj()
        elif path_type == 'path_enc_vec':
            return self.encode_paths(cutoff=cut_off)
        elif path_type == 'path_enc_aware_vec':
            return self.encode_position_aware_paths(seq_len=seq_len)
        else:
            raise NotImplemented('This method does not implement!')

    def gen_reduction_cell(self):
        reduction_cell = []
        for i in range(NUM_VERTICES):
            ops = np.random.choice(range(len(OPS)), 2)
            # input nodes for reduce
            nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)
            reduction_cell.extend([(nodes_in_reduce[0], ops[0]), (nodes_in_reduce[1], ops[1])])
        return reduction_cell

    @classmethod
    def generate_all_arch_back(cls):
        normal = []
        reduction = []
        for i in range(NUM_VERTICES):
            ops = np.random.choice(range(len(OPS)), NUM_VERTICES)

            # input nodes for conv
            nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
            # input nodes for reduce
            nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)

            normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
            reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])
        return (normal, reduction)

    def gen_outer_combination(self):
        num_ops = len(OPS)
        outer1 = [[(i, k), (j, l)] for i, j in [(0, 1), (1, 0)] for k in range(num_ops) for l in range(num_ops)]
        outer2 = [[(i, k), (j, l)] for i, j in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)] for k in range(num_ops) for l in range(num_ops)]
        outer = [[*o1, *o2] for o1 in outer1 for o2 in outer2]
        # print(len(outer1), len(outer2), len(outer))
        return outer

    def generate_normal_archs(self, save_path=''):
        num_ops = len(OPS)

        total_archs = 2*1*num_ops**2*3*2*num_ops**2*4*3*num_ops**2*5*4*num_ops**2
        total_archs_new = 2*1*2*1*num_ops**4*3*2*3*2*num_ops**4*4*3*4*3*num_ops**4*5*4*5*4*num_ops**4
        print(total_archs, total_archs_new)

        # models = {sha256(str([(i_1, k_1), (j_1, l_1), (i_2, k_2), (j_2, l_2), (i_3, k_3), (j_3, l_3), (i_4, j_4), (k_4, l_4)]).encode('utf-8')).hexdigest():
        #               [(i_1, k_1), (j_1, l_1), (i_2, k_2), (j_2, l_2), (i_3, k_3), (j_3, l_3), (i_4, j_4), (k_4, l_4)]
        #                 for i_1 in range(2) for j_1 in range(2) for k_1 in range(num_ops) for l_1 in range(num_ops)
        #                     for i_2 in range(3) for j_2 in range(3) for k_2 in range(num_ops) for l_2 in range(num_ops)
        #                         for i_3 in range(4) for j_3 in range(4) for k_3 in range(num_ops) for l_3 in range(num_ops)
        #                             for i_4 in range(5) for j_4 in range(5) for k_4 in range(num_ops) for l_4 in range(num_ops)}

        # i_1, j_1: (0, 1), (1, 0)
        # i_2, j_2: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)

        outer_idxs = 0
        outer_combination = self.gen_outer_combination()
        (i_1, k_1), (j_1, l_1), (i_2, k_2), (j_2, l_2) = outer_combination[outer_idxs]

        # for i_1 in range(2):
        #     j_1_idxes = list(range(2))
        #     j_1_idxes.remove(i_1)
        #     for j_1 in j_1_idxes:
        #         for k_1 in range(num_ops):
        #             for l_1 in range(num_ops):
        #                 for i_2 in range(3):
        #                     j_2_idxes = list(range(3))
        #                     j_2_idxes.remove(i_2)
        #                     for j_2 in j_2_idxes:
        #                         for k_2 in range(num_ops):
        #                             for l_2 in range(num_ops):
        for i_3 in range(4):
            j_3_idxes = list(range(4))
            j_3_idxes.remove(i_3)
            for j_3 in j_3_idxes:
                for k_3 in range(num_ops):
                    for l_3 in range(num_ops):
                        for i_4 in range(5):
                            j_4_idxes = list(range(5))
                            j_4_idxes.remove(i_4)
                            for j_4 in j_4_idxes:
                                for k_4 in range(num_ops):
                                    for l_4 in range(num_ops):
                                        normal_cell = [(i_1, k_1), (j_1, l_1), (i_2, k_2), (j_2, l_2), (i_3, k_3), (j_3, l_3), (i_4, k_4), (j_4, l_4)]
                                        reduction_cell = self.gen_reduction_cell()
                                        self.arch = (normal_cell, reduction_cell)
                                        key = sha256(str(self.arch).encode('utf-8')).hexdigest()
                                        # path_adj = self.get_path(path_type='adj_enc_vec', seq_len=612)
                                        # path_base_enc = self.get_path(path_type='path_enc_vec', seq_len=612)
                                        path_position_enc, (part1, part2) = self.get_path(path_type='path_enc_aware_vec', seq_len=612)
                                        file_name = os.path.join(save_path, key+'.pkl')
                                        with open(file_name, 'wb') as f:
                                            pickle.dump((normal_cell, reduction_cell), f)
                                            pickle.dump(key, f)
                                            # pickle.dump(path_adj, f)
                                            # pickle.dump(path_base_enc, f)
                                            pickle.dump(path_position_enc, f)
                                            pickle.dump(part1, f)
                                            pickle.dump(part2, f)
