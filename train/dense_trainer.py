#-*- coding: utf-8 -*-

import time
from collections import defaultdict

import numpy as np
import os
import torch
from torch import nn
from tqdm import tqdm
from utils import colorize
import pandas as pd
import json
import string
from datetime import datetime, timedelta

def seconds_to_string(seconds):
    """
    For writing subtitles 
    """
    if seconds<0:
        print('WARNING seconds is less than 0')
        seconds = max(seconds,0)
    timestr = '0' + str(timedelta(seconds=seconds))
    if len(timestr) > 8:
        if len(timestr) != 8+7:
            print('ERROR timestr ', timestr)
            print('seconds', seconds)
        assert len(timestr) == 8 + 7, 'timestring '+timestr
        timestr = timestr[:-3]
    else:
        timestr = timestr + '.000'
    assert len(timestr) == 12, 'timestring '+timestr
    return timestr


import pickle 

from train.base_trainer import BaseTrainer

class DenseTrainer(BaseTrainer):

    def train(self, dataloader, mode='train', epoch=-1):

        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        self.dataloader = dataloader

        tb_stepmod = (100 if mode == 'train' else 1) if not self.opts.test_only else 1

        bs = self.opts.batch_size
        counter = 0

        # cummulative losses and metrics dictionary
        metrics_dict = defaultdict( float)  

        bar = tqdm(total=len(dataloader))

        self.data_tic = self.step_tic = None
                
        if mode == 'test': 
            annotations = pickle.load(open(self.opts.test_labels_loc, 'rb'))
            dict_test_out = {}
            counts_dict = {}
        
        for b_id, batch_sample in enumerate(dataloader): 
            try:

                self.model.zero_grad()
                model_out = self.model.forward(batch_sample)
                # import pdb; pdb.set_trace()

                ### in text mode,  evaluate query data
                if mode == 'test':
                    for ib, vid_name in enumerate(batch_sample['video_id']): 
                        if vid_name not in dict_test_out.keys(): 
                            ### full length of video
                            dict_test_out[vid_name] = np.zeros(batch_sample['feat_len_video'][ib])
                            counts_dict[vid_name] = np.zeros(batch_sample['feat_len_video'][ib])
                        start_end_frames = [int(batch_sample['start_end_frames'][0][ib]), int(batch_sample['start_end_frames'][1][ib])]
                        dict_test_out[vid_name][start_end_frames[0]:start_end_frames[1]] += model_out['preds'][ib]
                        counts_dict[vid_name][start_end_frames[0]:start_end_frames[1]] += 1

                # ------------------------- Time steps  -------------------------

                if self.step_tic:
                    metrics_dict['t'] += time.time() - self.step_tic
                    if self.data_tic is not None:
                        metrics_dict['dt'] += time.time() - self.data_tic
                        metrics_dict['dt/total'] = (
                            metrics_dict['dt'] / metrics_dict['t']) * (counter + 1)
                self.step_tic = time.time()

                # ------------------------- Loss  -------------------------
                loss = model_out['loss'].mean()

                # ------------------------- Backprop  -------------------------
                if mode == 'train':
                    loss.backward(retain_graph=False)

                    if self.opts.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm(self.model.parameters(),
                                                    self.opts.grad_clip_norm)

                    self.optimizer.step()

                # ------------------------- Metrics  -------------------------

                metrics_dict['loss'] += model_out['loss'].mean().detach().cpu().item()

                # - tb summaries
                if (self.opts.test_only or
                        mode == 'train' and ( b_id % tb_stepmod) == 0
                            or b_id == len(dataloader) - 1):
                    for loss_name, loss_val in metrics_dict.items():
                        self.tb_writer.add_scalar(f'{mode}/{loss_name}',
                                                loss_val / (counter + 1),
                                                self.global_step)

                counter += 1
                if mode == 'train' or self.opts.test_only:
                    self.global_step += 1

                bar.update(1)

                desc = "%s: " % mode
                for cuml_name, cuml in sorted(metrics_dict.items()):
                    desc += "%s %.2f " % (cuml_name, cuml / counter)
                bar.set_description(desc)

                self.data_tic = time.time(
                )  # this counts how long we are waiting for data
            except:
                print('ERROR for ', vid_name)

        bar.close()

        if mode=='test':
            print('Evaluating...')
            for vid in dict_test_out.keys(): 
                
                counts_zeros = np.where(counts_dict[vid]==0)[0]
                if self.opts.print_text_output=="NO SIGNING": ### change all those with 0 counts to 1
                    dict_test_out[vid][counts_zeros] = 1
                counts_dict[vid][counts_zeros] = 1
                dict_test_out[vid] = dict_test_out[vid]/counts_dict[vid]
                dict_test_out[vid] = dict_test_out[vid]>= self.opts.threshold
                dict_test_out[vid][0] = 1
                dict_test_out[vid][-1] = 1

                ### make vtt
                no_signing = np.where(dict_test_out[vid]==1)[0]
                start_nosign = 0
                intervals = []
                for ix in range(len(no_signing)-1): 
                    if no_signing[ix] + 1 == no_signing[ix+1]:
                        pass
                    else: 
                        end_nosign = no_signing[ix]/25
                        intervals.append([start_nosign, end_nosign])
                        start_nosign = no_signing[ix+1]/25
                end_nosign = no_signing[-1]/25
                intervals.append([start_nosign, end_nosign])
            
                fw = open(f'output/{self.opts.save_output_folder}/' + vid + '.vtt', 'w')
                fw.write('WEBVTT\n\n')

                for ss, ee in intervals: 
                    if ss != ee:

                        fr_sec = float(ss)
                        to_sec = float(ee)

                        str_fr = seconds_to_string(fr_sec)
                        str_to = seconds_to_string(to_sec)
                        time_string = f'{str_fr} --> {str_to}'

                        fw.write(f'{time_string}\n')
                        fw.write(f'{self.opts.print_text_output}\n\n')

        desc = "Epoch end: %s: " % mode
        for cuml_name, cuml in sorted(metrics_dict.items()):
            desc += "%s %.2f " % (cuml_name, cuml / counter)
            self.tb_writer.add_scalar(f'{mode}_epoch/{cuml_name}',
                                      cuml / counter, self.global_step)
        print(desc)
        self.tb_writer.flush()

        return bar.desc 