# Self-supervised Representation Learning for Evolutionary Neural Architecture Search

This repository contains code for paper [Self-supervised Representation Learning for Evolutionary Neural Architecture Search]().

If you use the code please cite our paper.

    @article{Wei2020SSNENAS,
        title={Self-supervised Representation Learning for Evolutionary Neural Architecture Search},
        author={Chen Wei and Ji-min Liang},
        journal={ArXiv},
        year={2020},
        volume={abs/}
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
### Clone this project
```bash
git clone https://github.com/auroua/SSNENAS
cd SSNENAS
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