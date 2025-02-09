import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .combineNet import CNN_Network

def load_model(args,device):
    model = CNN_Network(args)
    # GPU settings
    if args.use_gpu:
        model.to(device)
        
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
        print("Weight pth loaded: ",args.resume)
    return model
