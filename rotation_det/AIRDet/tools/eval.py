#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import argparse
import os
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from airdet.config.base import parse_config
from airdet.apis.detector_inference import inference
from airdet.utils import fuse_model, get_local_rank, get_model_info, setup_logger, get_num_devices, synchronize
from airdet.dataset import make_data_loader
from airdet.detectors.detector_base import build_local_model, build_ddp_model

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_parser():
    parser = argparse.ArgumentParser("airdet eval")

    # distributed
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "-f",
        "--config_file",
        default=None,
        type=str,
        help="pls input your config file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    ) # TODO
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()

    device = "cuda"
    config = parse_config(args.config_file)
    config.merge(args.opts)

    file_name = os.path.join(config.miscs.output_dir, config.exp_name)

    if args.local_rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=args.local_rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        config.testing.conf_threshold = args.conf
    if args.nms is not None:
        config.testing.nms_iou_threshold = args.nms

    model = build_local_model(config, device)
    logger.info("Model Summary: {}".format(get_model_info(model, (640, 640))))

    model = build_ddp_model(model, local_rank=args.local_rank)

    model.cuda(args.local_rank)
    model.eval()

    ckpt_file = args.ckpt
    logger.info("loading checkpoint from {}".format(ckpt_file))
    loc = "cuda:{}".format(args.local_rank)
    ckpt = torch.load(ckpt_file, map_location=loc)
    new_state_dict = {}
    for k, v in ckpt['model'].items():
        k = "module." + k
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    # start evaluate
    output_folders = [None] * len(config.dataset.val_ann)


    if args.local_rank == 0 and config.miscs.output_dir:
        for idx, dataset_name in enumerate(config.dataset.val_ann):
            output_folder = os.path.join(config.miscs.output_dir, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    val_loader = make_data_loader(config, is_train=False)

    for output_folder, dataset_name, data_loader_val in zip(output_folders, config.dataset.val_ann, val_loader):
        inference(
            config,
            model,
            data_loader_val,
            dataset_name,
            iou_types = ("bbox",),
            box_only = False,
            device = device,
            output_folder = output_folder,
        )



if __name__ == "__main__":
    main()
