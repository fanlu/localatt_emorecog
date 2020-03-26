#!/usr/bin/env python
import os
import json as js
import argparse as argp
import pickle as pk
from tqdm import tqdm

import torch as tc
import torch.nn as nn

from torch.utils.data import DataLoader

from localatt.get_class_weight import get_class_weight
from localatt.lld_dataset import lld_dataset,lld_dataset2,lld_dataset3
from localatt.lld_dataset import lld_collate_fn,lld_collate_fn_2
from localatt.validate_lazy import validate_war_lazy
from localatt.validate_lazy import validate_uar_lazy
from localatt.localatt import localatt
from localatt.train import train
from localatt.train import validate_loop_lazy


if __name__ == "__main__":
    pars = argp.ArgumentParser()
    pars.add_argument('--propjs', help='property json')
    pars.add_argument('--feat_type', default='opensmile', help='feat_type')
    pars.add_argument('--feat_dim', type=int, default=32, help='feat_dim')
    pars.add_argument('--batch-size', type=int, default=128, help='batch_size')
    pars.add_argument('--epoch', type=int, default=200, help='epoch')
    pars.add_argument('--model', type=str, default='kefu/exp_20190409/model.pth', help='model will be loaded')
    pars.add_argument('--dataset', type=str, default='kefu', help='model will be loaded')
    args = pars.parse_args()
    device=tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
    with open(args.propjs) as f:
        #p = js.load(f.read())
        p = js.load(f)

    print(js.dumps(p, indent=4))
    #import pdb;pdb.set_trace()
    #with open(p['fulldata'], 'rb') as f:
    #    full_utt_lld = pk.load(f)

    #train_utt_lab_path    = p['dataset']+'/train_utt_lab.list'
    #dev_utt_lab_path      = p['dataset']+'/dev_utt_lab.list'
    #eval_utt_lab_path      = p['dataset']+'/eval_utt_lab.list'
    train_utt_lab_path    = args.dataset+'/wav_cat_train.list'
    dev_utt_lab_path      = args.dataset+'/wav_cat_dev.list'
    eval_utt_lab_path      = args.dataset+'/wav_cat_test.list'

    model_pth    =  args.model
    log          =  p['log']

    lr=p['lr']
    ephs=args.epoch
    bsz=args.batch_size

    #with open(p['dataset']+'/idx_label.json') as f:
    #    tgt_cls = js.load(f) # target classes

    featdim=args.feat_dim
    nhid=p['nhid']
    measure=p['measure']
    ncell=p['ncell']

    #nout = len(tgt_cls)
    #nout = 4

    cls_wgt = get_class_weight(train_utt_lab_path) # done.
    print('class weight:', cls_wgt)
    nout = len(cls_wgt)

    valid_lazy = {'uar': validate_uar_lazy,
                    'war': validate_war_lazy } # done.

    # loading

    #trainset = lld_dataset(full_utt_lld, train_utt_lab_path, cls_wgt, device)
    #devset = lld_dataset(full_utt_lld, dev_utt_lab_path, cls_wgt, device)
    #evalset = lld_dataset(full_utt_lld, eval_utt_lab_path, cls_wgt, device)
    #trainset = lld_dataset2(train_utt_lab_path, cls_wgt, device)
    #devset = lld_dataset2(dev_utt_lab_path, cls_wgt, device)
    #evalset = lld_dataset2(eval_utt_lab_path, cls_wgt, device)
    trainset = lld_dataset3(train_utt_lab_path, cls_wgt, normalize=True, feat_type=args.feat_type)
    devset = lld_dataset3(dev_utt_lab_path, cls_wgt, normalize=True, feat_type=args.feat_type)
    evalset = lld_dataset3(eval_utt_lab_path, cls_wgt, normalize=True, feat_type=args.feat_type)

    _collate_fn = lld_collate_fn(device=device) # done.
    trainloader = DataLoader(trainset, bsz, collate_fn=lld_collate_fn_2, shuffle=True, num_workers=10)
    devloader = DataLoader(devset, bsz, collate_fn=lld_collate_fn_2, num_workers=10)
    evalloader = DataLoader(evalset, bsz, collate_fn=lld_collate_fn_2, num_workers=10)
    # training
    model = localatt(featdim, nhid, ncell, nout) # done.
    model.to(device)
    if model_pth:
        model.load_state_dict(tc.load(model_pth))
    print(model)
    cls_wgt = tc.FloatTensor(cls_wgt).to(device)
    crit = nn.CrossEntropyLoss(weight=cls_wgt)

    optim = tc.optim.Adam(model.parameters(), lr=0.00005)

    _val_lz = valid_lazy[measure](crit=crit)
    _val_loop_lz = validate_loop_lazy(name='valid',
            loader=devloader,log=log)

    #trained = train(model, trainloader,
    #        _val_lz, _val_loop_lz, crit, optim, ephs, log)
    import pdb;pdb.set_trace()
    best_valid_score = 0.0
    best_model = model
    for epoch in tqdm(range(ephs), total=ephs):
        for i, batch in enumerate(trainloader):
            xs, lens = batch[0]
            inputs = xs.cuda(), lens.cuda()
            targets = batch[1].cuda()

            optim.zero_grad()
            model.train() # autograd on

            train_loss = crit(model(inputs), targets)
            train_loss.backward()

            optim.step()
            model.eval() # autograd off

            __val_lz = _val_lz(model=model)

        command ='[train] %4d/%4dth epoch, loss: %.3f'%(
                epoch, ephs, train_loss.item())
        os.system('echo "%s" >> %s'%(command, log))
        #import pdb;pdb.set_trace()
        valid_loss, valid_score, valid_score_p, valid_score_f1 = _val_loop_lz(__validate=__val_lz)

        if valid_score > best_valid_score:

            best_valid_score = valid_score

            command = '[valid] bestscore: %.3f, loss: %.3f, precision_score: %.3f, f1_score: %.3f'%(
                    valid_score, valid_loss, valid_score_p, valid_score_f1)
            os.system('echo "%s" >> %s'%(command, log))
            best_model = model

    print('Finished Training')

    tc.save(best_model.state_dict(), model_pth)

    model.load_state_dict(tc.load(model_pth))
    _val_lz = valid_lazy[measure](model=model, crit=crit)

    # testing
    validate_loop_lazy('test', _val_lz, evalloader, log)
