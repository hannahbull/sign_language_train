import os
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scipy import io
import json 
import pickle

import torch

class FeaturesDataset(Dataset): 
    
    def __init__(self, mode, opts):
        self.mode = mode
        self.opts = opts

        if self.opts.load_vocab:
            self.vocab_file = pickle.load(open(self.opts.vocab_file_loc, 'rb'))

        ### Build data dictionary
        self.data_dict = {}
        self.data_dict["txt"] = []
        self.data_dict["txt_idx"] = []
        self.data_dict["feats"] = []
        self.data_dict["feats_mask"] = []
        self.data_dict["video_id"] = []

        ### Load list of train/val/test split videos
        if (self.mode == 'train'): 
            ids = self.opts.train_ids
            data_path_root = self.opts.train_data_loc
            labels_path_root = self.opts.train_labels_loc
        elif (self.mode == 'val'): 
            ids = self.opts.val_ids
            data_path_root = self.opts.val_data_loc
            labels_path_root = self.opts.val_labels_loc
        elif (self.mode == 'test'): 
            ids = self.opts.test_ids
            data_path_root = self.opts.test_data_loc
            labels_path_root = self.opts.test_labels_loc
        else: 
            print('Choose mode = "train" or "val" or "test"')
        

        ### random subset of data (good for debugging)
        data_paths = os.listdir(data_path_root)
        if self.opts.random_subset_data < len(ids): 
            random.seed(123)
            data_paths = data_paths[0:self.opts.random_subset_data]
            print('Number of videos ', len(data_paths), data_paths)

        ### read ids
        video_ids = open(ids, "r").read().split('\n')

        ### load subtitle texts
        print('Loading labels... ')
        labels = pickle.load(open(labels_path_root, 'rb'))

        ### read features
        print('Loading features... ')
        for v in tqdm(video_ids): 
            self.data_dict['video_id'].append(v)

            labs = labels[v]

            feats = np.load(os.path.join(data_path_root, v + '.npy'))
            if len(feats) < self.opts.pad_input:
                feats = np.concatenate((feats, np.zeros((self.opts.pad_input - len(feats), feats.shape[1]))))
                feats_mask = np.concatenate((np.ones(len(feats)), np.zeros(self.opts.pad_input - len(feats))))
            elif len(feats) > self.opts.pad_input:
                feats = feats[0:self.opts.pad_input, :]
                feats_mask = np.ones(self.opts.pad_input)
            else: 
                feats_mask = np.ones(self.opts.pad_input)

            self.data_dict['feats'].append(feats)
            self.data_dict['feats_mask'].append(feats_mask)
            
            import pdb; pdb.set_trace()

    def __len__(self): 
        return len(self.data_dict["video_id"])

    def __getitem__(self, index):

        out_dict = {}
        out_dict["video_id"] = self.data_dict["video"][index]
        out_dict["txt_idx"] = self.data_dict["txt_idx"][index]
        out_dict["txt"] = self.data_dict["txt"][index]
        out_dict["feats"] = self.data_dict["feats"][index]
        out_dict["feats_mask"] = self.data_dict["feats_mask"][index]

        return out_dict
