import os
import sys
import argparse
sys.path.append(os.getcwd())
import pickle
import numpy as np


def parse_single_darts_macro_graph(single_model):
    with open(single_model, 'rb') as f:
        genotype = pickle.load(f)
        if isinstance(genotype, dict):
            return genotype
        else:
            models = pickle.load(f)
            hash_key = pickle.load(f)
            train_loss = pickle.load(f)
            val_acc = pickle.load(f)
            test_acc = pickle.load(f)
            best_val_acc = pickle.load(f)
            loss_list = pickle.load(f)
            val_acc_list = pickle.load(f)
            return [genotype, models, hash_key, train_loss, val_acc, test_acc, best_val_acc, loss_list, val_acc_list]


def parse_darts_macro_graph(models_path):
    models = os.listdir(models_path)
    full_models = [os.path.join(models_path, m) for m in models]
    best_acc = 0
    hash_key = None
    total_accs = []
    total_avg_accs = []
    total_model_keys = []
    for fm in full_models:
        data = parse_single_darts_macro_graph(fm)
        if isinstance(data, dict):
            test_acc = data['test_acc']
            total_accs.append(test_acc)
            total_avg_accs.append(data['val_acc'])
            hash_key = data['key']
            total_model_keys.append(hash_key)
        else:
            test_acc = data[6]
            avg_val_acc = sum(data[8][-5:])/len(data[8][-5:])
            avg_val_acc = (test_acc + avg_val_acc)/2
            total_accs.append(test_acc)
            total_avg_accs.append(avg_val_acc)
            total_model_keys.append(data[2])
            hash_key = data[2]
        if test_acc > best_acc:
            best_acc = test_acc
    idxs = np.argsort(np.array(total_avg_accs)).tolist()[::-1]
    print([total_accs[k] for k in idxs[:10]])
    print([total_avg_accs[k] for k in idxs[:10]])
    print([total_model_keys[k] for k in idxs[:10]])
    # print(best_acc, hash_key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for rank searched architectures.')
    parser.add_argument('--model_path', type=str,
                        default='/home/aurora/data_disk_new/train_output_2021/darts_open_search/model_pkl/',
                        help='darts')
    args = parser.parse_args()

    parse_darts_macro_graph(args.model_path)