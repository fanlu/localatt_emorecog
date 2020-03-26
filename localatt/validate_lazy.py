#xxx forward_lazy.py

from toolz import curry
from sklearn.metrics import recall_score, precision_score, f1_score

import logging
import tqdm

@curry
def validate_war_lazy(batch, model, crit):
    #import pdb;pdb.set_trace()
    xs, lens, keys = batch[0] # expected padded
    #xs, lens = xs.cuda(), lens.cuda()
    targets = batch[1] #.cuda()
    #sample_wgts = [x.cpu().item() for x in batch[2]]
    sample_wgts = batch[2]
    outputs = model(xs, lens)
    loss = crit(outputs, targets).data
    #import pdb;pdb.set_trace()
    recall_s = recall_score(
        outputs.max(dim=1)[1].data.cpu().numpy(),
        targets.data.cpu().numpy(),
        average='weighted',
        sample_weight=sample_wgts)
    precision_s = precision_score(
        outputs.max(dim=1)[1].data.cpu().numpy(),
        targets.data.cpu().numpy(),
        average='weighted',
        sample_weight=sample_wgts)
    f1_s = f1_score(
        outputs.max(dim=1)[1].data.cpu().numpy(),
        targets.data.cpu().numpy(),
        average='weighted',
        sample_weight=sample_wgts)

    return loss, recall_s, precision_s, f1_s

@curry
def validate_uar_lazy(batch, model, crit):
    inputs = batch[0] # expected padded
    targets = batch[1]
    sample_wgts = batch[2]

    outputs = model(inputs, lens)
    loss = crit(outputs, targets)

    score = recall_score(
        outputs.max(dim=1)[1].data.cpu().numpy(),
        targets.data.cpu().numpy(),
        average='macro')

    return loss, score

