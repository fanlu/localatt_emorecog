#xxx train.py

import os
from toolz import curry
from tqdm import tqdm
from torch.autograd import Variable
import torch

# on pytorch

@curry
def validate_loop_lazy(name, __validate, loader, log):

    losses = [0.0] * len(loader)
    scores = [0.0] * len(loader)
    p_scores = [0.0] * len(loader)
    f1_scores = [0.0] * len(loader)

    for i, batch in enumerate(tqdm(loader, total=len(loader))):

        losses[i], scores[i], p_scores[i], f1_scores[i] = __validate(batch)

    if len(loader) > 1:
        score = sum(scores[:-1])/(len(scores[:-1]))
        loss = sum(losses[:-1])/(len(losses[:-1]))
        p_score = sum(p_scores[:-1])/(len(p_scores[:-1]))
        f1_score = sum(f1_scores[:-1])/(len(f1_scores[:-1]))

    else:
        score = scores[0]
        loss = losses[0]
        p_score = p_scores[0]
        f1_score = f1_scores[0]

    command = '[%s] score: %.3f, loss: %.3f, p_score: %.3f, f1_score: %.3f'%(name, score, loss, p_score, f1_score)
    print(command)
    os.system('echo "%s" >> %s'%(command, log))

    return loss, score, p_score, f1_score


def train(model, loader, _valid_lazy, valid_loop, crit, optim, ephs, log, device=None):

    best_valid_score = 0.0
    best_model = model

    import pdb;pdb.set_trace()
    for epoch in tqdm(range(ephs), total=ephs):
        for i, batch in enumerate(loader):


            #xs,lens = batch[0]
            #inputs = xs.cuda(), lens.cuda()
            inputs = batch[0]
            targets = batch[1]

            optim.zero_grad()
            model.train() # autograd on

            train_loss = crit(model(inputs), targets)
            train_loss.backward()

            optim.step()
            model.eval() # autograd off

            __val_lz = _valid_lazy(model=model)

        command ='[train] %4d/%4dth epoch, loss: %.3f'%(
                epoch, ephs, train_loss.item())
        os.system('echo "%s" >> %s'%(command, log))

        valid_loss, valid_score = valid_loop(__validate=__val_lz)

        if valid_score > best_valid_score:

            best_valid_score = valid_score

            command = '[valid] bestscore: %.3f, loss: %.3f'%(
                    valid_score, valid_loss)
            os.system('echo "%s" >> %s'%(command, log))
            best_model = model

    print('Finished Training')

    return best_model

