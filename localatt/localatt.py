#xxx prognet.py

import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

def init_linear(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


# model = localatt(nin, nhid, ncell, nout)
class localatt(nn.Module):
    def __init__(self, featdim, nhid, ncell, nout):
        super(localatt, self).__init__()

        self.featdim = featdim
        self.nhid = nhid
        self.fc1 = nn.Linear(featdim, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.do2 = nn.Dropout()


        self.blstm = tc.nn.LSTM(nhid, ncell, 1,
                batch_first=True,
                dropout=0.5,
                bias=True,
                bidirectional=True)

        self.u = nn.Parameter(tc.zeros((ncell*2,)))
        # self.u = Variable(tc.zeros((ncell*2,)))

        self.fc3 = nn.Linear(ncell*2, nout)

        self.apply(init_linear)

    def forward(self, inputs, lens):

        #inputs = Variable(inputs_lens_tuple[0])
        batch_size = inputs.size()[0]
        #lens = list(inputs_lens_tuple[1])

        indep_feats = inputs.view(-1, self.featdim) # reshape(batch)

        indep_feats = F.relu(self.fc1(indep_feats))

        indep_feats = F.relu(self.do2(self.fc2(indep_feats)))

        batched_feats = indep_feats.view(batch_size, -1, self.nhid)

        packed = pack_padded_sequence(batched_feats, lens, batch_first=True)

        output, hn = self.blstm(packed)
        #print("lstm result shape:", output[0].shape)
        padded, lens = pad_packed_sequence(output, batch_first=True, padding_value=0.0)
        batch_size, seq_length, dim = padded.shape # bs * seq_l * dim
        #e = tc.matmul(padded, self.u)
        output = padded.contiguous().view(batch_size*seq_length, dim)
        #print("u shape:", self.u.unsqueeze(0).shape)
        output = F.linear(output, self.u.unsqueeze(0)).view(batch_size, seq_length)
        #print("alpha shape pre", output.shape)
        alpha = F.softmax(output, dim=-1)
        #print("alpha shape", alpha.shape)
        #m = tc.matmul(alpha, padded)
        #print("m left shape", alpha.unsqueeze(1).shape, "left shape", padded.shape)
        m = tc.bmm(alpha.unsqueeze(1), padded).squeeze(1)
        #print("m shape", m.shape)
        j = self.fc3(m)
        #print("result shape", j.shape)
        return F.softmax(j, dim=-1)

if __name__ == "__main__":
    example = tc.rand(4, 12, 13)
    model = localatt(13, 512, 128, 2)
    model.load_state_dict(tc.load('../kefu/exp_20190411/model.pth', map_location='cpu'))
    model.eval()
    print(model)
    output = model(example, tc.tensor([12]*4))
    output_1 = model(example[:2], tc.tensor([12]*2))
    print(tc.allclose(output[:2], output_1))
