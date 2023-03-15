import argparse
import datetime
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from timm.utils import AverageMeter, accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset  # For custom datasets
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, flop_count_str

from config import get_config
from data import build_loader
from models import build_model
from optimizer import build_optimizer
from utils import create_logger, load_checkpoint, save_checkpoint

import matplotlib as plt
from data.datasets import MediumImagenetHDF5Dataset
import random
from torchvision.utils import save_image
import matplotlib.pyplot as mpimg

def parse_option():
    parser = argparse.ArgumentParser("Vision model training and evaluation script", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs="+")
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--vis", type=int, help="number of images to visualize")


    args = parser.parse_args()
    config = get_config(args)
    return args, config

if __name__ == "__main__":
    args, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    # Make output dir
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.yaml")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test = build_loader(
        config
    )
    
    for i in range(vars(args)["vis"]):
        im = dataset_train.__getitem__(random.randint(0, dataset_train.__len__()))
        label = im[1]
        image = im[0]
        image.reshape(3,32,32).permute(1, 2, 0)
        image_path = "output/resnet18/image" + str(i) + "_" + str(label.item()) + ".png"
        # save image
        tensor  = image.cpu()
        save_image(tensor, image_path)
        # show image
        image = mpimg.imread(image_path)
        mpimg.title = image_path
        mpimg.imshow(image)
        mpimg.show()