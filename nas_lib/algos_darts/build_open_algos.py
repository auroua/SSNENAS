import sys
from .gin_predictor_open_search import gin_predictor_search_open
from .gin_predictor_open_search_fixed_nums import gin_predictor_fixed_num_search_open


def build_open_algos(agent):
    return getattr(sys.modules[__name__], agent+'_search_open')