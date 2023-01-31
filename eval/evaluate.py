### ground truth is the nouns in the subtitle
### task is to predict all the nouns
### metrics are F1, precision, recall

import torch
from torchmetrics import F1Score
target = torch.tensor([0, 1, 2, 0, 1, 2], ignore_index=0) # samplewise, global
preds = torch.tensor([0, 2, 1, 0, 0, 1])
f1 = F1Score(num_classes=3)
f1(preds, target)


pred_words = [52,36,12]
gt_nouns = [52,12,48,95]
