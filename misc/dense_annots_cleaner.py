from glob import glob
import os
import json
import webvtt
import scipy.io
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pickle

### This script makes binary annotations based off the manual annotations for pointing and for no signing
### The annotations come from the manual dense annotation (*NS and *P), as well as from the manually aligned subtitles
### in the case of no signing, in which case the parts without a manually aligned subtitle are deemed to be no signing. 

### Manual dense annotations: /scratch/shared/beegfs/hbull/repos/via-utils/saved_outputs_json/
### Manual subtitle alignment: /scratch/shared/beegfs/shared-datasets/bobsl/public_dataset_release_v1_2/subtitles/manually-aligned
### Outputs saved at: pointing_annotations/pointing.pkl, no_signing_annotations/no_signing.pkl
### The outputs are pickles with binary vectors of the same length as the number of frames (features) in the video
### 0 means not pointing or not 'no signing', 1 means pointing or no signing. 

### CHOOSE EITHER 'pointing' or 'no_signing'
task = 'pointing' # 'no_signing'

def get_gt_anns(jf):
	with open(jf) as jf:
		verified_anns = json.load(jf)
		fname_times = verified_anns["file"]
		assert len(fname_times) == 1
		fname_times = os.path.basename(fname_times["1"]["src"])
		fname = fname_times.split("#")[0]
		start, end = map(float, fname_times.split("#")[1][2:].split(','))

	print(f"Comparing annotations for {fname} from {start}s to {end}s")

	gt_anns = []
	for _, ann in verified_anns["metadata"].items():
		if ann["z"][0] >= start - 1 and ann["z"][0] <= end + 1:
			a = ann["av"]
			assert len(a) == 1
			word = a["1"]
			if word == "": continue
			
			if len(gt_anns) != 0 and word == gt_anns[-1][-1]:
				gt_anns[-1][1] = ann["z"][1]
			else:
				gt_anns.append([ann["z"][0], ann["z"][1], word])

	print(f"Segment contains {len(gt_anns)} GT annotations.")
	return fname, start, end, gt_anns

### LOAD MANUAL ANNOTATIONS
fnames, starts, ends, gts, preds = [], [], [], [], []
for jf in glob("/scratch/shared/beegfs/hbull/repos/via-utils/saved_outputs_json/*.json"):
	fname, start, end, gt = get_gt_anns(jf)
	fnames.append(fname)
	starts.append(start)
	ends.append(end)
	gts.append(gt)

annots_dict = {}
for ixf, f in enumerate(fnames): 
	if task == "no_signing":
		annots_gts = [[g[0], g[1]] for g in gts[ixf] if '*NS' or 'no sign' in g[2]]
	if task == "pointing":
		annots_gts = [[g[0], g[1]] for g in gts[ixf] if '*P' in g[2]]
	annots_dict[f] = annots_gts

### If the task is no_signing, then we use the manually aligned subtitles and take the spaces between these subtitles as 'no signing'
if task == "no_signing":
	subs_folder = "/scratch/shared/beegfs/shared-datasets/bobsl/public_dataset_release_v1_2/subtitles/manually-aligned"
	subs_dict = {}
	for f in fnames: 
		vtt_fname = f.replace('.mp4', '.vtt')
		subs = webvtt.read(os.path.join(subs_folder, vtt_fname))
		subs_start = []
		subs_end = []
		subs_text = []
		for s in subs: 
			if '[' not in s.text: 
				subs_start.append(s._start)
				subs_end.append(s._end)
				subs_text.append(s.text)
		
		subs_dict[f] = [[subs_start[s], subs_end[s]] for s in range(len(subs_start))]

### load features (just to get length of features)
def load_feat(episode):
	ep_features = scipy.io.loadmat(f"/scratch/shared/beegfs/shared-datasets/bsltrain/features/bobsl/featurise_c8697_pltp1_0.5_a_d8hasentsyn_m8prajhasent_swin-s_pretkinetics-v0-stride0.0625/filtered/{episode}/features.mat")["preds"]
	return episode, ep_features

video_ids = [v.replace('.mp4', '') for v in fnames]
with Pool(40) as p:
	df_collection = list(
		tqdm(p.imap(load_feat, video_ids), total=len(video_ids))
	)

feats_dict = {}
for i in df_collection: 
	feats_dict[i[0]] = i[1]

annots_binary = {}
for vid_id in video_ids: 
	binary_vec = np.zeros(len(feats_dict[vid_id]))
	if task == "no_signing":
		binary_vec = np.ones(len(feats_dict[vid_id]))
		for sub_interval in subs_dict[vid_id+'.mp4']:
			binary_vec[int(round(sub_interval[0]*25)):int(round(sub_interval[1]*25))] = 0 # no sign when there is not a subtitle
	for dense_annot_int in annots_dict[vid_id+'.mp4']: 
		binary_vec[int(round(dense_annot_int[0]*25)):int(round(dense_annot_int[1]*25))] = 1 # no sign/pointing when there is a *NS/*P annotation
	annots_binary[vid_id] = [int(b) for b in binary_vec]

if task == "no_signing":
	pickle.dump(annots_binary, open('no_signing_annotations/no_signing.pkl', 'wb'))
if task == "pointing": 
	pickle.dump(annots_binary, open('pointing_annotations/pointing.pkl', 'wb'))
