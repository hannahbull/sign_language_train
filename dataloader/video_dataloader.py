import os
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scipy import io
import json 

import torch

class VideoDataset(Dataset): 
    
    def __init__(self, mode, opts):
        self.mode = mode
        self.opts = opts


    def __len__(self): 
        return 

    def __getitem__(self, index):

        out_dict = {}

        return out_dict
