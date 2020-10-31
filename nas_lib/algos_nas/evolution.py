import random
import numpy as np
from nas_lib.data.nasbench_201_api.genotypes import Structure as CellStructure


# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix


def evolution_search_nasbench_101(search_space,
                                  num_init=10,
                                  k=10,
                                  population_size=30,
                                  total_queries=100,
                                  tournament_size=10,
                                  mutation_rate=1.0,
                                  allow_isomorphisms=False,
                                  deterministic=True,
                                  verbose=1,
                                  logger=None):
    """
    regularized evolution
    """
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    query = num_init
    val_losses = [d[4] for d in data]
    if num_init <= population_size:
        population = [i for i in range(num_init)]
    else:
        population = np.argsort(val_losses)[:population_size]

    while query <= total_queries:
        sample = random.sample(population, tournament_size)
        best_index = sorted([(i, val_losses[i]) for i in sample], key=lambda i: i[1])[0][0]
        mutated = search_space.mutate_arch({'matrix': data[best_index][1], 'ops': data[best_index][2]}, mutation_rate)
        data.append(mutated)
        val_losses.append(mutated[4])
        population.append(len(data) - 1)
        # kill the worst from the population   in nas bench paper kill the oldest arch
        if len(population) > population_size:
            worst_index = sorted([(i, val_losses[i]) for i in population], key=lambda i: i[1])[-1][0]
            population.remove(worst_index)

        if verbose and (query % k == 0):
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += 1
    return data


def evolution_search_nasbench_201(search_space,
                                  num_init=10,
                                  k=10,
                                  population_size=30,
                                  total_queries=100,
                                  tournament_size=10,
                                  mutation_rate=1.0,
                                  allow_isomorphisms=False,
                                  deterministic=True,
                                  verbose=1,
                                  logger=None):
    """
    regularized evolution
    """
    data = search_space.generate_random_dataset(num=num_init,
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    query = num_init
    val_losses = [d[4] for d in data]
    if num_init <= population_size:
        population = [i for i in range(num_init)]
    else:
        population = np.argsort(val_losses)[:population_size]

    while query <= total_queries:
        sample = random.sample(population, tournament_size)
        best_index = sorted([(i, val_losses[i]) for i in sample], key=lambda i: i[1])[0][0]
        arch = data[best_index][6]
        structures = CellStructure.str2structure(arch)
        _, mutated = search_space.mutate(structures)
        data.append(mutated)
        val_losses.append(mutated[4])
        population.append(len(data) - 1)
        # kill the worst from the population   in nas bench paper kill the oldest arch
        if len(population) > population_size:
            worst_index = sorted([(i, val_losses[i]) for i in population], key=lambda i: i[1])[-1][0]
            population.remove(worst_index)

        if verbose and (query % k == 0):
            top_5_loss = sorted([d[4] for d in data])[:min(5, len(data))]
            logger.info('Query {}, top 5 val losses {}'.format(query, top_5_loss))
        query += 1
    return data