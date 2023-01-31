# Pre-processing steps for extracting signers from diverse videos

See `youtube_vid_preprocessing/README.md`

# Training a simple model to recognise pointing and no signing

We train a simple transformer encoder model in order to predict whether there is pointing or no signing. 

The inputs are a temporal window of Swin features, and the outputs are a binary vector of the same length as the input features, where 1 denotes either pointing or no signing. 

The annotations come from manual dense annotations, as well as manual subtitle alignments for the case of no signing. In order to produce the annotation files, see `misc/dense_annots_cleaner.py`. 

The dataloader can be found here: `dataloader/features_dataloader_dense.py`. This loads the features and the annotations binary vector. 

The model can be found here: `models/encoders_joey.py`, copied from Joey NMT (https://github.com/joeynmt/joeynmt). 

The trainer can be found here: `train/dense_trainer.py`. It produces `.vtt` files which make a subtitle every time there is pointing or no signing. This is good for visualisation alongside the video. 

## Quick run

All of the config and their descriptions are here: `config/config.py`. 

To train the model to learn 'no signing' or 'pointing', run: 

```bash commands/train_nosigning.sh```
```bash commands/train_pointing.sh```

To evaluate the trained models for 'no signing' or 'pointing', run: 

```bash commands/test_nosigning.sh```
```bash commands/test_pointing.sh```

## Visualisation

A visualisation script to produce timelines and video crops around some of the outputs can be found here: 
`visualise_dense_annots/timeline_dense_annots.py`. 