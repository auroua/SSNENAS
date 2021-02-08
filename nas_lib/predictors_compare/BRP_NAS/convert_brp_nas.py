import copy

_opname_to_index = {
    'none': 0,
    'skip_connect': 1,
    'nor_conv_1x1': 2,
    'nor_conv_3x3': 3,
    'avg_pool_3x3': 4,
    'input': 5,
    'output': 6,
    'global': 7
}

_opindex_to_name = { value: key for key, value in _opname_to_index.items() }


def get_arch_vector_from_arch_str(arch_str):
    ''' Args:
            arch_str : a string representation of a cell architecture,
                for example '|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|'
    '''

    nodes = arch_str.split('+')
    nodes = [node[1:-1].split('|') for node in nodes]
    nodes = [[op_and_input.split('~')[0] for op_and_input in node] for node in nodes]

    # arch_vector is equivalent to a decision vector produced by autocaml when using Nasbench201 backend
    arch_vector = [_opname_to_index[op] for node in nodes for op in node]
    return arch_vector


def get_arch_str_from_arch_vector(arch_vector):
    ops = [_opindex_to_name[opindex] for opindex in arch_vector]
    return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*ops)