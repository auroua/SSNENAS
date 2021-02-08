#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from gnn_lib.data import Batch
from nas_lib.data.nasbench_101_torch import NASBenche101Dataset
from nas_lib.data.nasbench_201_torch import NASBenche201Dataset
from nas_lib.data.darts_torch import DartsDataset
from nas_lib.ccl.ccl_model.builder import build_model
from nas_lib.ccl.ccl_model.ccl_nas_model import CCLNas
from nas_lib.utils.comm import setup_logger, DummyLogger
from nas_lib.data.collate_batch import BatchCollator


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.gpu_count > 0:
        ngpus_per_node = args.gpu_count
    else:
        ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, distributed=False)


def main_worker(gpu, ngpus_per_node, args, distributed=True):
    args.gpu = gpu + args.gpu_base
    if args.multiprocessing_distributed:
        if args.gpu == args.gpu_base:
            file_name = 'log_%s_%d' % ('gpus', args.gpu)
            logger = setup_logger(file_name, args.save_dir, args.gpu, log_level='DEBUG',
                                  filename='%s.txt' % file_name)
        else:
            logger = DummyLogger()
    else:
        file_name = 'log_%s_%d' % ('gpus', args.gpu)
        logger = setup_logger(file_name, args.save_dir, args.gpu, log_level='DEBUG',
                              filename='%s.txt' % file_name)
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    model = CCLNas(
        build_model(args.arch, args.with_g_func), args.input_dim, args.moco_dim_fc,
        args.moco_dim, distributed=distributed, train_samples=args.train_samples, t=args.moco_t,
        min_negative_size=args.min_negative_size, margin=args.margin)
    logger.info(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(),
                                 args.lr,
                                 betas=(0.0, 0.9),
                                 weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.search_space == 'nasbench_101':
        train_dataset = NASBenche101Dataset(model_type='SS_CCL')
    elif args.search_space == 'nasbench_201':
        train_dataset = NASBenche201Dataset(model_type='SS_CCL')
    elif args.search_space == 'darts':
        train_dataset = DartsDataset(model_type='SS_CCL', arch_path=args.darts_arch_path)
    else:
        raise NotImplementedError('This kind nasbench has not implemented.')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    collator = BatchCollator()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True,
        collate_fn=collator)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        center_vec = train_nested(train_loader, model, criterion, optimizer, epoch, args, logger)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_path = os.path.join(args.save_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'centers': center_vec
            }, is_best=False, filename=save_path)


def train_nested(train_loader, model, criterion, optimizer, epoch, args, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch), logger=logger)

    # switch to train mode
    model.train()
    device = torch.device(f'cuda:{args.gpu}')
    end = time.time()
    center_list = []
    step = args.batch_step
    for i, (g_d, path_encodings) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        batch = Batch.from_data_list(g_d)
        batch = batch.to(device)

        indices = list(range(len(g_d)))
        random.shuffle(indices)
        indices = indices[:args.train_samples]
        indices_list = [indices[i*step:(i+1)*step] for i in range(args.train_samples//step)]
        for idxss, sample_ids in enumerate(indices_list):
            # compute output
            logits, label, centers = model(batch=batch, path_encoding=path_encodings,
                                           device=device, search_space=args.search_space,
                                           sample_ids=sample_ids,
                                           logger=logger)
            loss = criterion(logits, label)
            if args.center_regularization:
                center_dist = torch.mm(centers, centers.T)
                masks = torch.ones_like(center_dist)
                eigen_val = list(range(center_dist.size(0)))
                masks[eigen_val, eigen_val] = 0
                center_loss = 0.5*torch.mean(masks*center_dist)
                loss = loss + 0.5*center_loss
            size_logits = logits.size(0)
            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits, label, topk=(1, 5))
            losses.update(loss.item(), size_logits)
            top1.update(acc1[0], size_logits)
            top5.update(acc5[0], size_logits)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        center_list.append(centers.cpu().detach().numpy())
    return center_list


def assemble_training_data_batch(dist_matrix):
    min_val, min_indices = torch.min(dist_matrix, dim=1)
    nums = torch.sum(dist_matrix == min_val, dim=1)

    max_nums = torch.sum(dist_matrix > (min_val + 2), dim=1)
    nums_np = nums.numpy()
    nums_np.sort()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
