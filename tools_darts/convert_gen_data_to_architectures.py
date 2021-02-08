import os
import sys
sys.path.append(os.getcwd())
import argparse
from nas_lib.utils.comm import set_random_seed
from nas_lib.data.darts_init import gen_darts_dataset
from tools_darts.gen_darts_archs import Genotype


if __name__ == '__main__':
    # Running this code have to modify the predictor_list, search_space, trails, seq_len, gpu, load_dir, save_dir
    parser = argparse.ArgumentParser(description='Predictor comparison parameters!')
    parser.add_argument('--nums', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=112)
    parser.add_argument('--save_dir', type=str,
                        default='/home/aurora/data_disk_new/train_output_2021/darts_save_path/darts_partial.pkl')

    args = parser.parse_args()
    set_random_seed(args.seed)

    gen_darts_dataset(args.save_dir,
                      save_path=args.save_dir)