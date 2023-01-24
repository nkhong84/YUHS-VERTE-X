import cv2
import random
import os, sys
from glob import glob
import numpy as np

from .model import load_model
from .model.core import test_score

from .utils.data_loader import load_dataloader

from collections import defaultdict
import torch
import torch.nn as nn
import warnings

from glob import glob
import warnings
import gc
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings(action='ignore')

torch.backends.cudnn.benchmark = True

# Seed
RANDOM_SEED = 1111
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def main(args):
    
    # Argument
    args.stype = "osteoporosis"
    args.exp_pth = "exp"

    if args.resume != None:
        resume = sorted(glob(f"{exp_folder}/{stype}/*.pth"))[-1]
        args.resume = resume

    # GPU setting
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # dataloader
    dataloader = load_dataloader(args)

    # Model
    model = load_model(args,device)

    res = test_score(dataloader, device, model, args)
    

if __name__ == '__main__':
    main()