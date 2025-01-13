import cv2
import random
import os, sys
from glob import glob
import numpy as np

from .model import load_model
from .model.core import test_score

from .utils.data_loader import load_dataloader
from .utils.config import ParserArguments

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

def main(args):
   # If the resume argument is provided, load the latest pre-trained model
    if args.resume != None:
        # Find all .pth files in the specified experiment folder and sort them
        resume = sorted(glob(f"{exp_folder}/{stype}/*.pth"))[-1]
        args.resume = resume  # Set the most recent model checkpoint to args.resume

    # GPU setting: Use the specified GPU if available; otherwise, fallback to CPU
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Data loader: Initialize the data loader based on the provided arguments
    dataloader = load_dataloader(args)

    # Model loading: Initialize the model and load any pre-trained weights if specified
    model = load_model(args, device)

    # Model testing: Evaluate the model using the test dataset and return the results
    results = test_score(dataloader, device, model, args)

    # Final result
    # "The final results are stored as a dictionary, which can be processed using pandas or other tools as desired."


if __name__ == '__main__':
    main()