# Self-supervised Representation Learning for Evolutionary Neural Architecture Search

This repository contains code for paper [Self-supervised Representation Learning for Evolutionary Neural Architecture Search](https://arxiv.org/abs/2011.00186).

If you use the code please cite our paper.

    @article{Wei2020SSNENAS,
        title={Self-supervised Representation Learning for Evolutionary Neural Architecture Search},
        author={Chen Wei and Yiping Tang and Chuang Niu and Haihong Hu and Yue Wang and Jimin Liang},
        journal={ArXiv},
        year={2020},
        volume={abs/2011.00186}
    }

## Prerequisites
* Python 3.7
* Pytorch 1.3
* Tensorflow 1.14.0
* ptflops `pip install --upgrade git+https://github.com/sovrasov/flops-counter.pytorch.git`
* torch-scatter `pip install torch-scatter==1.4.0`
* torch-sparse `pip install torch-sparse==0.4.3`
* torch-cluster `pip install torch-cluster==1.4.5`
* torch-spline-conv `pip install torch-spline-conv==1.1.1`
* umap_learn `pip install umap_learn`

## Searching Environment
* Ubuntu 18.04
* cuda 10.0
* cudnn 7.5.1

## Usage
#### Clone this project
```bash
git clone https://github.com/auroua/SSNENAS
cd SSNENAS
```

#### Data Preparation
##### NASBench-101
1. Down load `NASBench-101` dataset first. We only use the `nasbench_only108.tfrecord` file.
2. Set the variable `nas_bench_101_base_path` in `config.py` to point to the folder containing the file `nasbench_only108.tfrecord`.
3. Run the following command to generate data files that are required by the code.
```bash
python nas_lib/data/nasbench_101_init.py
```

##### NASBench-201
1. Down load the `NASBench-201` dataset first. In this experiment, we use the `NASBench-201` dataset with version `v1_1-096897`, and the file name is `NAS-Bench-201-v1_1-096897.pth`.
2. Set the variable `nas_bench_201_base_path` in `config.py` to point to the folder containing the file `NAS-Bench-201-v1_1-096897.pth`.
3. Run the following command to generate data files that are required by the code.
```bash
python nas_lib/data/nasbench_201_init.py --dataname cifar10-valid
python nas_lib/data/nasbench_201_init.py --dataname cifar100
python nas_lib/data/nasbench_201_init.py --dataname ImageNet16-120
```


## Acknowledge
1. [bananas](https://github.com/naszilla/bananas)
2. [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
3. [NAS-Bench-101](https://github.com/google-research/nasbench)
4. [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201)
6. [MoCo](https://github.com/facebookresearch/moco)

## Contact
Chen Wei

email: weichen_3@stu.xidian.edu.cn, weichen@xupt.edu.cn