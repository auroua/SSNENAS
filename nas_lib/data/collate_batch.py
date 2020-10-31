import torch
import numpy as np


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        g_list = transposed_batch[0]
        path_encoding_list = np.array(transposed_batch[1])
        path_encoding = torch.Tensor(path_encoding_list)
        return g_list, path_encoding