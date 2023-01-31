# Pre-processing steps for extracting signers from diverse videos

## YouTube download

This uses `yt-dlp`: https://github.com/yt-dlp/yt-dlp 


```
import os 
from tqdm import tqdm

### List of videos to download:
video_list = ['Ax9Sy1VjUVQ', 'Pt_DR5aTCjw', 'wNfMbb_LUPA']

### Check what has already been downloaded
video_list_downloaded = os.listdir('video_download_dir/')

### Extract video IDs from file names
video_list_downloaded = [vid.split(' ')[-1].replace('[', '').replace(']', '').split('.')[0] for vid in video_list_downloaded]

### Download videos which have not yet been downloaded 
for vid in tqdm(video_list):
    if vid not in video_list_downloaded:
        os.system(f"timeout 100s yt-dlp -f 'bestvideo*' --write-subs --compat-options no-live-chat -P video_download_dir/  https://youtu.be/{vid}")

```

## Convert videos to .mp4 at 25 fps

This uses `ffmpeg`: https://ffmpeg.org/ 

The `-crf` argument controls quality, adapt as required.

```
import os 

video_list_downloaded = os.listdir('video_download_dir/')

for i, vid in enumerate(video_list_downloaded):
    try:
        name_orig = 'video_download_dir/' + vid

        # Change the video names to the YouTube id code
        vid_id = vid.split(' ')[-1].replace('[', '').replace(']', '').split('.')[0]
        name_dest = 'video_mp4_25fps/' + vid_id + '.mp4'

        cmd = f"ffmpeg -y -i \"{name_orig}\" -crf 23 -c:v libx264 -filter:v fps=fps=25 {name_dest}"
        os.system(cmd)
    except:
        print('error for', cmd)
```

## Make info file with frames, height and width of videos

```
import pandas as pd
import cv2 
import os
from tqdm import tqdm

def frame_count_hw(video_path, manual=False):
    def manual_count(handler):
        frames = 0
        while True:
            status, frame = handler.read()
            if not status:
                break
            frames += 1
        return frames 

    cap = cv2.VideoCapture(video_path)
    # Slow, inefficient but 100% accurate method 
    if manual:
        frames = manual_count(cap)
    # Fast, efficient but inaccurate method
    else:
        try:
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            if frames < 0:
                frames = manual_count(cap)
        except:
            frames = manual_count(cap)
            try: 
                cap.get(cv2.CAP_PROP_FPS)
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            except:
                fps = -1
                height = -1
                width = -1
    cap.release()
    return frames, fps, height, width

vid_dir = 'video_mp4_25fps'
save_loc = 'info.csv'

vid_ids = os.listdir(vid_dir)
vid_ids = [v.replace('.mp4', '') for v in vid_ids]

vid_list = []
frames_vids = []
fps_vids = []
height_vids = []
width_vids = []
for vid in tqdm(vid_ids):
    frames, fps, height, width = frame_count_hw(os.path.join(vid_dir,vid +'.mp4'))
    frames_vids.append(frames)
    fps_vids.append(fps)     
    height_vids.append(int(height))
    width_vids.append(int(width))
    vid_list.append(vid)

df = {'video_id': vid_list, 'fps': fps_vids, 'frames': frames_vids, 'height': height_vids, 'width': width_vids}
df = pd.DataFrame.from_dict(df)
df.to_csv(save_loc, index=False)
```

## [Not necessary] Get title and date of YouTube videos

```
import pandas as pd
import cv2 
import os
from tqdm import tqdm

vid_ids = os.listdir('video_mp4_25fps')
vid_ids = [v.replace('.mp4', '') for v in vid_ids]

vid_list = []
date_list = []
title_list = []
for vid in tqdm(vid_ids): 
    try:
        cmd = f'yt-dlp https://youtu.be/{vid} --print "%(upload_date)s %(title)s"'
        out = os.popen(cmd).read().strip()
        dateval = out.split(' ')[0]
        title = out.split(' ')[1:]
        title = ' '.join(title)
        vid_list.append(vid)
        date_list.append(dateval)
        title_list.append(title)
    except: 
        print('ERROR FOR ', vid)

df = {'video_id': vid_list, 'date': date_list, 'title': title_list}
df = pd.DataFrame.from_dict(df)
df.to_csv('date_title_info.csv', index=False)
```

## Extract OpenPose keypoints

Using OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose 

``` 
import os
import csv
import subprocess
import shutil

# videos dir
videos_dir = 'video_mp4_25fps'

# save openpose dir
openpose_dir = 'openpose'

list_all_videos = os.listdir(videos_dir)

os.makedirs("/tmp/videodir/", exist_ok=True)

list_done = os.listdir(openpose_dir)
for vid in list_all_videos:
    if vid + '.zip' not in list_done: 
        os.system("rm -r /tmp/videodir/*")
        new_vid_name = vid.replace(f'{videos_dir}/', '')
        print('Completing...' , vid)
        cmd = f"scp {videos_dir}/{vid} /tmp/videodir/"
        os.system(cmd)
        
        ### Command for my setup, otherwise just:
        ### cmd = f"openpose --video /tmp/videodir/{vid} --model_folder=/opt/openpose/models --display 0 --num_gpu 3 --render_pose 0 --face --hand --write_json /tmp/{new_vid_name}/"
        cmd = f"singularity exec --nv /scratch/shared/beegfs/hbull/repos/openpose.sif /opt/openpose/build/examples/openpose/openpose.bin --video /tmp/videodir/{vid} --model_folder=/opt/openpose/models --display 0 --num_gpu 3 --render_pose 0 --face --hand --write_json /tmp/{new_vid_name}/"
        os.system(cmd)
        shutil.make_archive(f'/tmp/{new_vid_name}', 'zip', f'/tmp/{new_vid_name}/')
        cmd = f"mv /tmp/{new_vid_name}.zip {openpose_dir}/{new_vid_name}.zip"
        os.system(cmd)
        cmd = f'rm -r /tmp/{new_vid_name}/'
        os.system(cmd)
        os.system("rm -r /tmp/OpenPose*")
    else: 
        print('Completed already ', vid)
```

## Extracting sequences of signers

This uses https://github.com/hannahbull/clean_op_data_sl

Change argument `--max_number_signers` to keep the most likely $N$ signers per video scene. 

```
import zipfile
import os
import pandas as pd
from tqdm import tqdm

from multiprocessing import Process, Manager
from multiprocessing import Pool

NUMBER_CPUS = 62

height_width = pd.read_csv('info.csv')

vid_ids = os.listdir('video_mp4_25fps')
vid_ids = [v.replace('.mp4', '') for v in vid_ids]

list_existing = os.listdir('signer_sequences') # output directory

vid_ids = [v for v in vid_ids if v not in list_existing]

def get_signer_sequences(vid):

    os.makedirs('tmp/'+vid, exist_ok=True)
    with zipfile.ZipFile(f"openpose/{vid}.mp4.zip", mode="r") as archive:
        archive.extractall('tmp/'+vid)
    
    height = int(height_width[height_width['video_id']==vid].height)
    width = int(height_width[height_width['video_id']==vid].width)

    cmd = f'python scripts/clean_op_data_sl/clean_op_data.py --height_video {height} --width_video {width} --openpose_folder tmp/{vid}/ --output_folder signer_sequences/{vid}/ --max_number_signers 2'
    os.system(cmd)

    cmd = f'rm -r tmp/{vid}'
    os.system(cmd)

with Pool(NUMBER_CPUS) as p:
    p.map(get_signer_sequences, vid_ids)
```

## Crop videos around most likely signer

This uses https://github.com/hannahbull/clean_op_data_sl

This extracts the portion of video surrounding one subtitle. Requires subtitles saved in directory with same name as videos. 

```
import webvtt
import os
import numpy as np 
import pickle
import pandas as pd
import sys
from clean_op_data_sl.utils import compute_size_movement

from multiprocessing import Process, Manager
from multiprocessing import Pool

def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

NUMBER_CPUS = 62

### load csv with columns 'video_id', 'height' (in pixels), 'width' (in pixels), 'frames' (number of frames)
info = pd.read_csv('info.csv')

vid_ids = info['video_id'].tolist()
heights = info['height'].tolist()
widths = info['width'].tolist()
len_vids = info['frames'].tolist()

vid_numbers = [v for v in range(len(vid_ids))]

def save_video_extracts(i):
    try:
        vid = vid_ids[i]
        height = heights[i]
        width = widths[i]
        len_vid = len_vids[i]

        ### load subtitles
        subs = webvtt.read('subtitles/'+ vid + '.fr.vtt') 

        ### load signer sequences
        list_seqs = os.listdir('signer_sequences/' + vid)
        starts_seq = [int(l.split('_')[0]) for l in list_seqs]
        ends_seq = [int(l.split('_')[0]) + int(l.split('_')[1])  for l in list_seqs]

        ### sort starts and ends
        ends_seq = [x for _, x in sorted(zip(starts_seq, ends_seq))]
        starts_seq = sorted(starts_seq)

        intervals = [item for t in zip(starts_seq, ends_seq) for item in t]

        for sub_no, sub in enumerate(subs): 
            if not os.path.isfile(f'video_mp4_25fps/{vid}_{str(sub_no).zfill(4)}.mp4'):

                startframe = int(round(sub._start*25))
                endframe = int(round(sub._end*25))
                midframe = (startframe+endframe)//2

                # find interval containing midpoint
                overlap = np.searchsorted(intervals, midframe)//2

                # make sure overlap is above 90%
                percent_overlap = getOverlap([starts_seq[overlap], ends_seq[overlap]], [startframe, endframe])/(endframe-startframe)

                if percent_overlap > 0.9:
                    ### read file
                    file_name_sequence = 'signer_sequences/'+vid + '/' + str(starts_seq[overlap]) + '_' + str(ends_seq[overlap]-starts_seq[overlap])+'_data.pkl'
                    file_pickle = pickle.load(open(file_name_sequence, 'rb'))
                    ### extract part of file
                    if startframe >= file_pickle[0][0]: ## if startframe is in pickle
                        start_idx = np.where(file_pickle[0]==startframe)[0][0]
                    else:
                        start_idx = 0
                        startframe = file_pickle[0][0]
                    if endframe <= file_pickle[0][-1]:
                        end_idx = np.where(file_pickle[0]==endframe)[0][0] + 1
                    else: 
                        end_idx = len(file_pickle[0])
                        endframe = file_pickle[0][-1] + 1
                    
                    start_time_s = max(0,startframe/25 - 0.5)
                    end_time_s = min(endframe/25 + 0.5, len_vid/25)
                    duration_s = end_time_s - start_time_s

                    ### body keypoints
                    body = file_pickle[2][start_idx:end_idx]

                    ### most likely signer
                    signer_stats = []
                    for j in range(body.shape[-1]):
                        height_stat, movemement_stat = compute_size_movement(body[:,:,:,j:(j+1)])
                        signer_stats.append(height_stat*movemement_stat)
                    
                    which_signer = np.argmax(np.array(signer_stats))

                    body_x = body[:,0,:,which_signer].flatten()
                    body_y = body[:,1,:,which_signer].flatten()
                    body_conf = body[:,2,:,which_signer].flatten()

                    body_x = body_x[body_conf>0]
                    body_y = body_y[body_conf>0]

                    max_x = min(np.percentile(body_x, 99) + 0.05,1)*width
                    max_y = min(np.percentile(body_y, 99) + 0.05,1)*height

                    min_x = max(0,np.percentile(body_x, 1) - 0.05)*width
                    min_y = max(0,np.percentile(body_y, 1) - 0.15)*height

                    dim_square = int(min(height, max((max_x-min_x), (max_y-min_y))))

                    max_x_square = int(min(width, (max_x+min_x)/2 + dim_square/2))
                    min_x_square = max_x_square - dim_square

                    max_y_square = int(min(height, (max_y+min_y)/2 + dim_square/2))
                    min_y_square = max_y_square - dim_square

                    cmd1 = f'ffmpeg -y -i video_mp4_25fps/{vid}.mp4 -ss {start_time_s} -t {duration_s} -async 1 -pix_fmt yuv420p -threads 1 video_mp4_25fps_cut/{vid}_{str(sub_no).zfill(4)}.mp4'
                    cmd2 = f'ffmpeg -y -i video_mp4_25fps_cut/{vid}_{str(sub_no).zfill(4)}.mp4 -filter:v "crop={dim_square}:{dim_square}:{min_x_square}:{min_y_square},scale=444:444" -pix_fmt yuv420p -threads 1 video_mp4_25fps_crops/{vid}_{str(sub_no).zfill(4)}.mp4'

                    os.system(cmd1) # crops video
                    os.system(cmd2) # resizes video to 444x444 pixels
            else:
                print('Video exists, skipping')
    except:
        pass

with Pool(NUMBER_CPUS) as p:
    p.map(save_video_extracts, vid_numbers)
```

## Get sign segmentations

See https://github.com/RenzKa/sign-segmentation 

```
python demo/demo.py --video_path video_mp4_25fps_crops/Ax9Sy1VjUVQ_0002.mp4 --save_segments --generate_vtt --save_path sign_segmentation
```

## Extract features

See https://github.com/gulvarol/bsl1k 

## Extract parts of speech of subtitles

```
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from tqdm import tqdm

import os
import pickle
import webvtt 

tokenizer = AutoTokenizer.from_pretrained("gilf/french-postag-model")
model = AutoModelForTokenClassification.from_pretrained("gilf/french-postag-model")

nlp_token_class = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)

vid_list = os.listdir('subtitles') 
vid_list = [v.replace('.fr.vtt', '') for v in vid_list]

sub_dict = {}

for vid in tqdm(vid_list): 

    subs = webvtt.read('subtitles/'+ vid + '.fr.vtt')

    for ixsub, sub in enumerate(subs): 
        text = sub.text
        start_time = sub._start
        end_time = sub._end
        
        tokens = nlp_token_class(text)

        word_list = []
        score_list = []
        pos_list = []
        for t in tokens: 
            word_list.append(t['word'])
            score_list.append(t['score'])
            pos_list.append(t['entity_group'])
    
        k = f'{vid}_{str(ixsub).zfill(4)}'
        
        sub_dict[k] = {'text': text, 
                        'start_time': start_time, 
                        'end_time': end_time, 
                        'pos_list': pos_list, 
                        'text_list': word_list, 
                        'score_list': score_list}

pickle.dump(sub_dict, open('subtitles.pkl', 'wb'))
```

## Make vocab file with nouns only
```
import pickle
from tqdm import tqdm
import re

subtitles_pkl = pickle.load(open('videos/subtitles.pkl', 'rb'))

def remove_plural(w):
    w_singl = w
    if len(w_singl)>3 and w_singl[-3:] == 'aux' and w_singl[0].lower()==w_singl[0] and w_singl[-4:] != 'eaux':
        w_singl = re.sub('aux$', 'al', w_singl)
    if len(w_singl)>1:
        w_singl = w_singl if w_singl[-1]!='s' else w_singl[:-1]
        w_singl = w_singl if w_singl[-1]!='x' else w_singl[:-1]
    return w_singl

vocab_dict = {}

for vid_id in tqdm(subtitles_pkl.keys()):
    for iw, w in enumerate(subtitles_pkl[vid_id]['text_list']): 
        if subtitles_pkl[vid_id]['pos_list'][iw] in ['NC', 'NPP']: 
            w_singl = remove_plural(w)
            w_singl = re.sub('^’ ', '', w_singl)
            w_singl = re.sub(' d$', '', w_singl)
            w_singl = re.sub(' n$', '', w_singl)
            w_singl = re.sub('“', '', w_singl)
            w_singl = re.sub('…', '', w_singl)
            w_singl = re.sub('«', '', w_singl)
            w_singl = re.sub('»', '', w_singl)
            w_singl = re.sub('‘', '', w_singl)
            w_singl = re.sub('’', '', w_singl)
            w_singl = re.sub('”', '', w_singl)
            w_singl = re.sub('–', '', w_singl)
            w_singl = re.sub('—', '', w_singl)
            w_singl = re.sub('^\' ', '', w_singl)
            w_singl = re.sub('\\*', '', w_singl)
            w_singl = re.sub('\\+', '', w_singl)
            w_singl = re.sub('\\-', '', w_singl)
            w_singl = re.sub('\\.', '', w_singl)
            w_singl = re.sub('\\%', '', w_singl)
            w_singl = re.sub('\\&', '', w_singl)
            w_singl = re.sub('\\@', '', w_singl)
            w_singl = w_singl.strip()
            for w_split in w_singl.split(' '):
                w_split = remove_plural(w_split)
                if w_split not in vocab_dict.keys():
                    vocab_dict[w_split] = 0
                vocab_dict[w_split] += 1

list_words = list(vocab_dict.keys())
for w in list_words: 
    if vocab_dict[w] < 5 or '#' in w or w in ['', '\'']:
        vocab_dict.pop(w)

vocab_list = list(vocab_dict.keys())
vocab_list.sort()

vocab_list = ['<PAD>', '<MASK>', '<OOV>'] + vocab_list
print(vocab_list, 'len', len(vocab_list))

words_to_idx = {}
idx_to_words = {}
count = 0
for w in vocab_list: 
    words_to_idx[w] = count
    idx_to_words[count] = w
    count+=1

vocab = {'words_to_idx': words_to_idx, 'idx_to_words': idx_to_words}

pickle.dump(vocab, open('videos/vocab.pkl', 'wb'))
```