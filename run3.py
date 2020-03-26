#!/usr/bin/env python
from localatt.train import validate_loop_lazy
from localatt.train import train
from localatt.localatt import localatt
from localatt.validate_lazy import validate_uar_lazy
from localatt.validate_lazy import validate_war_lazy
from localatt.lld_dataset import lld_collate_fn, lld_collate_fn_2
from localatt.lld_dataset import lld_dataset, lld_dataset2, lld_dataset3
from localatt.get_class_weight import get_class_weight
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import torch.nn as nn
import torch as tc
import os
import json as js
import argparse as argp
import pickle as pk
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
import logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def validate(loader, model, crit, classes, on_gpu):

    losses = [0.0] * len(loader)
    predicts = np.array([])
    trues = np.array([])
    sample_wgts = []
    for i, batch in enumerate(loader):
        #import pdb;pdb.set_trace()
        xs, lens, keys = batch[0] # expected padded
        #xs, lens = xs.cuda(), lens.cuda()
        targets = batch[1] #.cuda()
        #sample_wgts = [x.cpu().item() for x in batch[2]]
        sample_wgts += list(batch[2])
        if on_gpu:
            xs, lens, targets = xs.cuda(), lens.cuda(), targets.cuda()
        outputs = model(xs, lens)
        if crit:
            losses[i] = crit(outputs, targets).data
        predicts = np.append(predicts, outputs.max(dim=1)[1].data.cpu().numpy())
        trues = np.append(trues, targets.data.cpu().numpy())
    # logging.info(trues)
    # logging.info(predicts)
    t = classification_report(trues, predicts, sample_weight=sample_wgts)
    logging.info(t)
        # losses[i], scores[i], p_scores[i], f1_scores[i] = validate_war(batch, model, crit, on_gpu)

    if len(loader) > 1:
        loss = sum(losses[:-1])/(len(losses[:-1]))
    else:
        loss = losses[0]

    logging.info('valid loss: %.3f'%(loss))

    return loss

if __name__ == "__main__":
    pars = argp.ArgumentParser()
    pars.add_argument('--propjs', help='property json')
    pars.add_argument('--feat_type', default='opensmile', help='feat_type')
    pars.add_argument('--feat_dim', type=int, default=32, help='feat_dim')
    pars.add_argument('--batch-size', type=int, default=128, help='batch_size')
    pars.add_argument('--epoch', type=int, default=200, help='epoch')
    pars.add_argument('--exp_dir', type=str, default=None,
                      required=True, help='model path will be saveed or loaded')
    # pars.add_argument('--model_pth', type=str, default=None, required=True, help='model path will be saveed or loaded')
    pars.add_argument('--restore', type=str2bool, default=False,
                      help='model will be loaded')
    pars.add_argument('--eval', type=str2bool, default=False,
                      help='only eval,do not train')
    pars.add_argument('--gpu', type=str2bool, default=True, help='use gpu')
    pars.add_argument('--dataset', type=str, default='kefu',
                      help='dataset will be loaded')
    pars.add_argument('--test_dataset', type=str,
                      default='kefu', help='test dataset will be loaded')
    pars.add_argument('--lr', type=float, default=5e-05, help='learning rate')
    pars.add_argument('--nhid', type=int, default=512, help='nhid')
    pars.add_argument('--ncell', type=int, default=128, help='ncell')
    pars.add_argument('--measure', type=str,
                      default='war', help='measure metric')
    args = pars.parse_args()
    if args.gpu:
        import kaldi
        kaldi.SelectGpuId('yes')
        kaldi.CuDeviceAllowMultithreading()
        device = tc.device('cuda')
        num_workers = 10
    else:
        device = tc.device('cpu')
        num_workers = 0

    train_utt_lab_path = args.dataset+'/feats.scp'
    train_lab_path = args.dataset + '/utt2spk'
    dev_utt_lab_path = args.test_dataset+'/feats.scp'
    dev_lab_path = args.test_dataset+'/utt2spk'
    eval_utt_lab_path = args.test_dataset+'/feats.scp'
    eval_lab_path = args.test_dataset+'/utt2spk'

    model_pth = f'{args.exp_dir}/best.pth'
    lr = args.lr
    ephs = args.epoch
    bsz = args.batch_size
    featdim = args.feat_dim
    nhid = args.nhid
    measure = args.measure
    ncell = args.ncell

    cls_wgt, classes = get_class_weight(train_lab_path)  # done.
    logging.info(f'class weight:{cls_wgt}, {classes}')
    nout = len(cls_wgt)

    valid_lazy = {'uar': validate_uar_lazy,
                  'war': validate_war_lazy}  # done.

    trainset = lld_dataset3(train_utt_lab_path, cls_wgt, classes,
                            train_lab_path, normalize=True, feat_type=args.feat_type)
    devset = lld_dataset3(dev_utt_lab_path, cls_wgt, classes,
                          dev_lab_path, normalize=True, feat_type=args.feat_type)
    evalset = lld_dataset3(eval_utt_lab_path, cls_wgt, classes,
                           eval_lab_path, normalize=True, feat_type=args.feat_type)

    _collate_fn = lld_collate_fn(device=device)  # done.
    trainloader = DataLoader(
        trainset, bsz, collate_fn=lld_collate_fn_2, shuffle=True, num_workers=num_workers)
    devloader = DataLoader(
        devset, bsz, collate_fn=lld_collate_fn_2, num_workers=num_workers)
    evalloader = DataLoader(
        evalset, bsz, collate_fn=lld_collate_fn_2, num_workers=num_workers)
    # training
    model = localatt(featdim, nhid, ncell, nout)  # done.
    model.to(device)
    if args.restore:
        model.load_state_dict(tc.load(model_pth))
    logging.info(model)
    cls_wgt = tc.FloatTensor(cls_wgt).to(device)
    crit = nn.CrossEntropyLoss(weight=cls_wgt)

    optim = tc.optim.Adam(model.parameters(), lr=0.00005)

    _val_lz = valid_lazy[measure](crit=crit)
    # _val_loop_lz = validate_loop_lazy(name='valid',
    #                                   loader=devloader, log=log)

    # trained = train(model, trainloader,
    #        _val_lz, _val_loop_lz, crit, optim, ephs, log)
    # import pdb;pdb.set_trace()
    best_valid_score = 0.0
    best_model = model
    if not args.eval:
        for epoch in tqdm(range(ephs), total=ephs):
            for i, batch in enumerate(trainloader):
                (xs, lens, keys), targets, wgts = batch
                if args.gpu:
                    xs, lens = xs.cuda(), lens.cuda()
                    targets = targets.cuda()

                optim.zero_grad()
                model.train()  # autograd on

                train_loss = crit(model(xs, lens), targets)
                train_loss.backward()

                optim.step()
                model.eval()  # autograd off

                # __val_lz = _val_lz(model=model)

            logging.info('[train] %4d/%4dth epoch, loss: %.3f' % (
                epoch, ephs, train_loss.item()))
            # os.system('echo "%s" >> %s' % (command, log))
            #import pdb;pdb.set_trace()
            # valid_loss, valid_score, valid_score_p, valid_score_f1 = _val_loop_lz(
            #     __validate=__val_lz)
            # validation
            valid_score = validate(devloader, model, crit, classes, args.gpu)

            if valid_score > best_valid_score:

                best_valid_score = valid_score
                # os.system('echo "%s" >> %s' % (command, log))
                best_model = model
                tc.save(best_model.state_dict(), "%s.%s" % (model_pth, epoch))

        print('Finished Training')

        tc.save(best_model.state_dict(), model_pth)
    else:
        m = tc.load(model_pth, map_location='cpu')
        model.load_state_dict(m)
        model.to('cpu')
        model.eval()
        # print(model)
        validate(evalloader, model, None, classes, False)
        # with open('/mnt/cephfs2/asr/users/fanlu/kaldi2/egs/m8/data/test_16/predict_py_5_%s.scp' % bsz, 'w') as f:
        #     for i, batch in enumerate(evalloader):
        #         if i > 30:
        #             break
        #         xs, lens, keys = batch[0]
        #         #xs, lens = xs.cuda(), lens.cuda()
        #         #import pdb;pdb.set_trace()
        #         start = time.time()
        #         outputs = model(xs, lens)
        #         t = time.time()-start
        #         print("input shape %s, consume %s s" % (xs.shape, t))
        #         #output_1 = model(xs[:10], lens[:10])
        #         #print(tc.allclose(outputs[:10], output_1))
        #         label = outputs.max(dim=1)[1].data.cpu().numpy()
        #         for key, l in zip(keys, label):
        #             f.write("%s %s\n" % (key, l))

