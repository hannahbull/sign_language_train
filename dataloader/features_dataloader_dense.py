import os
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from multiprocessing import Pool
from tqdm import tqdm

import scipy.io
import json 
import pickle

import torch

class DenseFeaturesDataset(Dataset): 
    """ This dataloader loads dense sequences of features with their frame-level annotations
    """
    
    def __init__(self, mode, opts):
        self.mode = mode
        self.opts = opts

        if self.opts.load_vocab:
            self.vocab_file = pickle.load(open(self.opts.vocab_file_loc, 'rb'))

        ### Build data dictionary
        self.data_dict = {}
        self.data_dict["txt_idx"] = []
        self.data_dict["feats"] = []
        self.data_dict["feats_mask"] = []
        self.data_dict["video_id"] = []
        self.data_dict["start_end_frames"] = []
        self.data_dict["feat_len_video"] = []

        ### Load list of train/val/test split videos
        if (self.mode == 'train'): 
            self.ids = self.opts.train_ids
            self.data_path_root = self.opts.train_data_loc
            self.labels_path_root = self.opts.train_labels_loc
        elif (self.mode == 'val'): 
            self.ids = self.opts.val_ids
            self.data_path_root = self.opts.val_data_loc
            self.labels_path_root = self.opts.val_labels_loc
        elif (self.mode == 'test'): 
            self.ids = self.opts.test_ids
            self.data_path_root = self.opts.test_data_loc
            self.labels_path_root = self.opts.test_labels_loc
        else: 
            print('Choose mode = "train" or "val" or "test"')

        ### read ids
        if self.ids[-4:] == '.txt':
            video_ids = open(self.ids, "r").read().split('\n')
        else:
            video_ids = [str(self.ids)]
        
        ### random subset of data (good for debugging)
        if self.opts.random_subset_data < len(self.ids): 
            video_ids = video_ids[0:self.opts.random_subset_data]
            print('Number of videos ', len(video_ids), video_ids)

        ### load subtitle texts
        if not self.mode == "test":
            print('Loading annotations... ')
            annotations = pickle.load(open(self.labels_path_root, 'rb'))

        if 1: # os.path.exists(f"{self.data_path_root}/{video_ids[0]}/features.mat"):
            with Pool(self.opts.n_workers) as p:
                df_collection = list(
                    tqdm(p.imap(self.load_feat, video_ids), total=len(video_ids))
                )

            feats_dict = {}
            for i in df_collection:
                if i[1] is not None:
                    feats_dict[i[0]] = i[1]
            
            video_ids = list(feats_dict.keys())

        ### read features
        print('Loading features... ')
        for v in tqdm(video_ids): 

            if os.path.exists(os.path.join(self.data_path_root, v + '.npy')):
                feats = np.load(os.path.join(self.data_path_root, v + '.npy'))
            else: 
                feats = feats_dict[v]

            if not self.mode == 'test': # or v in annotations.keys():
                annots = np.array(annotations[v])
            else: 
                annots = np.zeros(len(feats))

            assert len(feats) == len(annots)

            ### every 50 at stride 25
            for i in np.arange(0,len(feats), self.opts.stride): 
                if not (self.opts.remove_labels_all_zero and np.max(annots[i:(i+self.opts.input_len)]) == 0) and (i+self.opts.input_len)<=len(feats):
                    self.data_dict['video_id'].append(v)
                    self.data_dict['feats'].append(feats[i:(i+self.opts.input_len)])
                    self.data_dict['txt_idx'].append(annots[i:(i+self.opts.input_len)]) 
                    self.data_dict['start_end_frames'].append([i,(i+self.opts.input_len)]) 
                    self.data_dict['feat_len_video'].append(len(feats))
            
    ### loading features 
    def load_feat(self, episode):
        try:
            ep_features = scipy.io.loadmat(f"{self.data_path_root}/{episode}/features.mat")["preds"]
            return episode, ep_features
        except:
            return episode, None

    def __len__(self): 
        return len(self.data_dict["video_id"])

    def __getitem__(self, index):

        out_dict = {}
        out_dict["video_id"] = self.data_dict["video_id"][index]
        out_dict["txt_idx"] = self.data_dict["txt_idx"][index]
        out_dict["feats"] = self.data_dict["feats"][index]
        out_dict["start_end_frames"] = self.data_dict["start_end_frames"][index]
        out_dict["feat_len_video"] = self.data_dict["feat_len_video"][index]

        return out_dict
