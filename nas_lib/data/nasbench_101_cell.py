import numpy as np
import copy
import nas_lib.data.nasbench_101_api.model_spec as api
from nas_lib.utils.utils_data import find_isolate_node
import random
import itertools as it
from collections import defaultdict
from nas_lib.algos import algo_sort

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
ISOLATE = 'isolate'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


class Cell:
    def __init__(self, matrix, ops, isolate_node_idxs=None):
        self.matrix = matrix
        self.ops = ops
        self.isolate_node_idxs = isolate_node_idxs

    def serialize(self):
        return {
            'matrix': self.matrix,
            'ops': self.ops
        }

    def modelspec(self):
        return api.ModelSpec(matrix=self.matrix, ops=self.ops)

    @classmethod
    def random_cell(cls, nasbench):
        """ 
        From the NASBench repository 
        https://github.com/google-research/nasbench
        """
        while True:
            matrix = np.random.choice(
                [0, 1], size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT
            spec = api.ModelSpec(matrix=matrix, ops=ops)
            if nasbench.is_valid(spec):
                return {
                    'matrix': matrix,
                    'ops': ops
                }

    @classmethod
    def random_cell_gnn(cls, nasbench):
        """
        From the NASBench repository
        https://github.com/google-research/nasbench
        """
        while True:
            matrix = np.random.choice(
                [0, 1], size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            isolate_nodes = find_isolate_node(matrix)
            ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT

            spec = api.ModelSpec(matrix=matrix, ops=ops)
            if nasbench.is_valid(spec):
                return {
                    'matrix': matrix,
                    'ops': ops,
                    'isolate_node_idxs': isolate_nodes
                }

    @classmethod
    def random_cell_both(cls, nasbench):
        """
        From the NASBench repository
        https://github.com/google-research/nasbench
        """
        while True:
            matrix = np.random.choice(
                [0, 1], size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            matrix_orig = matrix.copy()
            isolate_nodes = find_isolate_node(matrix)
            ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT

            spec = api.ModelSpec(matrix=matrix, ops=ops)
            if nasbench.is_valid(spec):
                return {
                    'matrix': matrix,
                    'matrix_orig': matrix_orig,
                    'ops': ops,
                    'isolate_node_idxs': isolate_nodes
                }

    def get_val_loss(self, nasbench, deterministic=1, patience=50):
        if not deterministic:
            # output one of the three validation accuracies at random
            return 100*(1 - nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy'])
        else:
            # query the api until we see all three accuracies, then average them
            # a few architectures only have two accuracies, so we use patience to avoid an infinite loop
            accs = []
            while len(accs) < 3 and patience > 0:
                patience -= 1
                acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']
                if acc not in accs:
                    accs.append(acc)
            return round(100*(1-np.mean(accs)), 3)

    def get_val_loss2(self, nasbench, deterministic=1, patience=50):
        if not deterministic:
            # output one of the three validation accuracies at random
            return 100*(nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy'])
        else:
            # query the api until we see all three accuracies, then average them
            # a few architectures only have two accuracies, so we use patience to avoid an infinite loop
            accs = []
            while len(accs) < 3 and patience > 0:
                patience -= 1
                acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']
                if acc not in accs:
                    accs.append(acc)
            return round(100*(np.mean(accs)), 3)

    def get_val_loss_nn_pred(self, nasbench, deterministic=1, patience=50):
        accs = []
        test_accs = []
        while len(accs) < 3 and patience > 0:
            patience -= 1
            acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']
            test_acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['test_accuracy']
            # if acc not in accs:
            accs.append(acc)
            # if test_acc not in test_accs:
            test_accs.append(test_acc)
        return 100*(np.mean(np.array(accs))), 100*(np.mean(np.array(test_accs)))

    def get_test_loss(self, nasbench, patience=50):
        """
        query the api until we see all three accuracies, then average them
        a few architectures only have two accuracies, so we use patience to avoid an infinite loop
        """
        accs = []
        while len(accs) < 3 and patience > 0:
            patience -= 1
            acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['test_accuracy']
            if acc not in accs:
                accs.append(acc)
        return round(100*(1-np.mean(accs)), 3)

    def get_test_loss2(self, nasbench, patience=50):
        """
        query the api until we see all three accuracies, then average them
        a few architectures only have two accuracies, so we use patience to avoid an infinite loop
        """
        accs = []
        while len(accs) < 3 and patience > 0:
            patience -= 1
            acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['test_accuracy']
            if acc not in accs:
                accs.append(acc)
        return 100*round((np.mean(accs)), 3)

    def perturb(self, nasbench, edits=1):
        """ 
        create new perturbed cell 
        inspird by https://github.com/google-research/nasbench
        """
        new_matrix = copy.deepcopy(self.matrix)
        new_ops = copy.deepcopy(self.ops)
        for _ in range(edits):
            while True:
                if np.random.random() < 0.5:
                    for src in range(0, NUM_VERTICES - 1):
                        for dst in range(src+1, NUM_VERTICES):
                            new_matrix[src][dst] = 1 - new_matrix[src][dst]
                else:
                    for ind in range(1, NUM_VERTICES - 1):
                        available = [op for op in OPS if op != new_ops[ind]]
                        new_ops[ind] = np.random.choice(available)

                new_spec = api.ModelSpec(new_matrix, new_ops)
                if nasbench.is_valid(new_spec):
                    break
        return {
            'matrix': new_matrix,
            'ops': new_ops
        }

    def mutate(self, nasbench, mutation_rate=1.0):
        """
        similar to perturb. A stochastic approach to perturbing the cell
        inspird by https://github.com/google-research/nasbench
        """
        while True:
            new_matrix = copy.deepcopy(self.matrix)
            new_ops = copy.deepcopy(self.ops)

            edge_mutation_prob = mutation_rate / NUM_VERTICES
            for src in range(0, NUM_VERTICES - 1):
                for dst in range(src + 1, NUM_VERTICES):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            op_mutation_prob = mutation_rate / OP_SPOTS
            for ind in range(1, OP_SPOTS + 1):
                if random.random() < op_mutation_prob:
                    available = [o for o in OPS if o != new_ops[ind]]
                    new_ops[ind] = random.choice(available)

            new_spec = api.ModelSpec(new_matrix, new_ops)
            if nasbench.is_valid(new_spec):
                return {
                    'matrix': new_matrix,
                    'ops': new_ops
                }

    def mutate2(self, nasbench, mutation_rate=1.0):
        """
        similar to perturb. A stochastic approach to perturbing the cell
        inspird by https://github.com/google-research/nasbench
        """
        iteration = 0
        while True:
            new_matrix = copy.deepcopy(self.matrix)
            new_ops = copy.deepcopy(self.ops)

            vertices = self.matrix.shape[0]
            op_spots = vertices - 2
            edge_mutation_prob = mutation_rate / vertices
            for src in range(0, vertices - 1):
                for dst in range(src + 1, vertices):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            if op_spots != 0:
                op_mutation_prob = mutation_rate / op_spots
                for ind in range(1, op_spots + 1):
                    if random.random() < op_mutation_prob:
                        available = [o for o in OPS if o != new_ops[ind]]
                        new_ops[ind] = random.choice(available)

            new_spec = api.ModelSpec(new_matrix, new_ops)
            ops_idx = [-1] + [OPS.index(new_ops[idx]) for idx in range(1, len(new_ops)-1)] + [-2]
            iteration += 1
            if iteration == 500:
                ops_idx = [-1] + [OPS.index(self.ops[idx]) for idx in range(1, len(self.ops) - 1)] + [-2]
                return {
                    'matrix': copy.deepcopy(self.matrix),
                    'ops': copy.deepcopy(self.ops),
                    'ops_idx': ops_idx
                }
            if nasbench.is_valid(new_spec):
                return {
                    'matrix': new_matrix,
                    'ops': new_ops,
                    'ops_idx': ops_idx
                }

    def mutate_rates(self, nasbench, edge_rate, node_rate):
        """
        similar to perturb. A stochastic approach to perturbing the cell
        inspird by https://github.com/google-research/nasbench
        """
        while True:
            new_matrix = copy.deepcopy(self.matrix)
            new_ops = copy.deepcopy(self.ops)
            h, w = new_matrix.shape
            edge_mutation_prob = edge_rate
            for src in range(0, h - 1):
                for dst in range(src + 1, h):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            op_mutation_prob = node_rate
            for ind in range(1, OP_SPOTS + 1):
                if random.random() < op_mutation_prob:
                    available = [o for o in OPS if o != new_ops[ind]]
                    new_ops[ind] = random.choice(available)

            new_spec = api.ModelSpec(new_matrix, new_ops)
            if nasbench.is_valid(new_spec):
                return {
                    'matrix': new_matrix,
                    'ops': new_ops
                }

    def encode_cell(self):
        """
        compute the "standard" encoding,
        i.e. adjacency matrix + op list encoding
        """
        OPS = [CONV3X3, CONV1X1, MAXPOOL3X3, ISOLATE]
        encoding_length = (NUM_VERTICES ** 2 - NUM_VERTICES) // 2 + OP_SPOTS * len(OPS)
        encoding = np.zeros((encoding_length))
        n = 0
        for i in range(NUM_VERTICES - 1):
            for j in range(i+1, NUM_VERTICES):
                encoding[n] = self.matrix[i][j]
                n += 1
        for i in range(1, NUM_VERTICES - 1):
            op_idx = OPS.index(self.ops[i])
            encoding[n+op_idx] = 1
            n += 4
        return tuple(encoding)

    def get_paths(self):
        """ 
        return all paths from input to output
        """
        paths = []
        for j in range(0, NUM_VERTICES):
            paths.append([[]]) if self.matrix[0][j] else paths.append([])
        # create paths sequentially
        for i in range(1, NUM_VERTICES - 1):
            for j in range(1, NUM_VERTICES):
                if self.matrix[i][j]:
                    for path in paths[i]:
                        paths[j].append([*path, self.ops[i]])
        return paths[-1]

    def get_path_indices(self):
        """
        compute the index of each path
        There are 3^0 + ... + 3^5 paths total.
        (Paths can be length 0 to 5, and for each path, for each node, there
        are three choices for the operation.)
        """
        paths = self.get_paths()
        mapping = {CONV3X3: 0, CONV1X1: 1, MAXPOOL3X3: 2}
        path_indices = []
        for path in paths:
            index = 0
            for i in range(NUM_VERTICES - 1):
                if i == len(path):
                    path_indices.append(index)
                    break
                else:
                    index += len(OPS) ** i * (mapping[path[i]] + 1)
        return tuple(path_indices)

    def encode_paths(self):
        """ output one-hot encoding of paths """
        num_paths = sum([len(OPS) ** i for i in range(OP_SPOTS + 1)])
        path_indices = self.get_path_indices()
        path_encoding = np.zeros(num_paths)
        for index in path_indices:
            path_encoding[index] = 1
        return path_encoding

    def path_distance(self, other):
        """ 
        compute the distance between two architectures
        by comparing their path encodings
        """
        return np.sum(np.array(self.encode_paths() != np.array(other.encode_paths())))

    def edit_distance(self, other):
        """
        compute the distance between two architectures
        by comparing their adjacency matrices and op lists
        """
        graph_dist = np.sum(np.array(self.matrix) != np.array(other.matrix))
        ops_dist = np.sum(np.array(self.ops) != np.array(other.ops))
        return graph_dist + ops_dist

    def verify_correctness(self, arch, data_list, edit_dist):
        dist_list = []
        new_matrix = arch[0]
        new_ops = arch[1]
        for d in data_list:
            arch = d[0]
            if isinstance(arch, tuple):
                graph_dist = np.sum(np.array(new_matrix) != np.array(arch[0]))
                ops_dist = np.sum(np.array(new_ops) != np.array(arch[1]))
                dist_list.append(graph_dist+ops_dist)
            elif isinstance(arch, dict):
                graph_dist = np.sum(np.array(new_matrix) != np.array(arch['matrix']))
                ops_dist = np.sum(np.array(new_ops) != np.array(arch['ops']))
                dist_list.append(graph_dist+ops_dist)
            else:
                raise NotImplementedError()
        flag = min(dist_list) >= edit_dist
        return flag

    def generate_edit_compose(self, edge_list, op_list, edit_distance):
        total_list = edge_list + op_list
        idxs = list(range(len(total_list)))
        return list(it.combinations(idxs, edit_distance))

    def mutate_edit_distance(self, nasbench, edit_dist, candidate_num, data):
        """
        similar to perturb. A stochastic approach to perturbing the cell
        inspird by https://github.com/google-research/nasbench
        """
        arch_list = []
        new_ops = copy.deepcopy(self.ops)
        edges = [(src, dst) for src in range(0, NUM_VERTICES - 1) for dst in range(src+1, NUM_VERTICES)]
        op_available_tuple = []
        for ind in range(1, OP_SPOTS + 1):
            available = [o for o in OPS if o != new_ops[ind]]
            for o in available:
                op_available_tuple.append((ind, o))

        idx_list = self.generate_edit_compose(edges, op_available_tuple, edit_dist)
        random.shuffle(idx_list)
        for edit_idx in idx_list:
            if edit_dist > 1 and len(arch_list) >= candidate_num:
                break
            new_matrix = copy.deepcopy(self.matrix)
            new_ops = copy.deepcopy(self.ops)

            for j in edit_idx:
                if j >= len(edges):
                    nest_idx = op_available_tuple[j - len(edges)]
                    new_ops[nest_idx[0]] = nest_idx[1]
                else:
                    edge_conn = edges[j]
                    new_matrix[edge_conn[0], edge_conn[1]] = 1 - new_matrix[edge_conn[0], edge_conn[1]]
            isolate_nodes = find_isolate_node(new_matrix)
            new_spec = api.ModelSpec(new_matrix, new_ops)
            flag = self.verify_correctness((new_matrix, new_ops), data, edit_dist)
            if nasbench.is_valid(new_spec) and flag:
                arch_list.append({
                    'matrix': new_matrix,
                    'ops': new_ops,
                    'isolate_node_idxs': isolate_nodes
                })
        return arch_list

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

    def get_paths_seq_aware(self):
        paths = []
        paths_idx = []
        for j in range(0, NUM_VERTICES):
            paths.append([[]]) if self.matrix[0][j] else paths.append([])
            paths_idx.append([[]]) if self.matrix[0][j] else paths_idx.append([])
        for i in range(1, NUM_VERTICES - 1):
            for j in range(1, NUM_VERTICES):
                if self.matrix[i][j]:
                    for ids, path in enumerate(paths[i]):
                        paths[j].append([*path, self.ops[i]])
                        paths_idx[j].append([*paths_idx[i][ids], i])
        return paths[-1], paths_idx[-1]

    def encode_paths_seq_aware(self, length):
        """ output one-hot encoding of paths """
        mapping = {CONV3X3: 0, CONV1X1: 1, MAXPOOL3X3: 2}
        paths, paths_idx = self.get_paths_seq_aware()
        paths, paths_idx = self.sort_seqs_list(paths, paths_idx)
        vectors_list = []
        for (p_list, idx_list) in zip(paths, paths_idx):
            vec = np.zeros(OP_SPOTS*len(OPS), dtype=np.int16)
            for p, ids in zip(p_list, idx_list):
                vec[(ids-1)*len(OPS) + mapping[p]] = 1
            vectors_list.append(vec)
        path_encoding = np.array(vectors_list, dtype=np.int16)
        path_encoding = path_encoding.reshape((1, -1))[0]
        residual_len = length - path_encoding.shape[0]
        if residual_len != 0:
            residual_np = np.zeros(residual_len, dtype=np.int16)
            path_encoding = np.concatenate([path_encoding, residual_np])
        return path_encoding

    def get_encoding(self, predictor_type, seq_len):
        if predictor_type == 'adj_enc_vec':
            return self.encode_cell()
        elif predictor_type == 'path_enc_vec':
            return self.encode_paths()
        elif predictor_type == 'path_enc_aware_vec':
            return self.encode_paths_seq_aware(seq_len)
        else:
            raise NotImplementedError()