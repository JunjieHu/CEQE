#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable
from vocab import Vocab, VocabEntry
import numpy as np

#################################################################################
# Utility functions

def tensor_transform(linear, X):
    # X is a 3D tensor, [n, T, D] -> [n, T, d] via linear function
    return linear(X.contiguous().view(-1, X.size(2))).view(X.size(0), X.size(1), -1)

def load_embedding(vocab, file, dim=100):
    hit = 0
    embed = np.random.rand(len(vocab), dim) * 2 - 1
    lower_vocab_dict = {k.lower():k for k in vocab.word2id}

    for lidx, line in enumerate(open(file, 'r')):
        if lidx == 0:
            vocab_size, dim = map(int, line.split(' '))
            continue
        line = line.strip()
        items = line.split(' ')
        #assert len(items) == dim + 1, "len={}, dim+1={}".format(len(items), dim+1)
        if len(items) != dim + 1:
            continue
        word = items[0].strip()
        if word in lower_vocab_dict:
            hit += 1
            wid = vocab[lower_vocab_dict[word]]
            embed[wid] = np.array([float(t) for t in items[1:]])
    print('Hit {}/{} words in vocabulary'.format(hit, len(vocab)))
    return embed

# x:    [n, d, T1]
# y:    [n, T2, d]
# align:[n, T1, T2]
# op: left, mid, right
def get_align_vector(x, y, align, op='left'):
    A = Variable(align.data.new(align.size()).zero_())
    y_ret = Variable(y.data.new(y.size()).zero_())
    if op == 'mid':
        A = align
        y_ret = y
    elif op == 'left' and y.size(1) > 1:
        A[:, :-1, :] = align[:, 1:, :]
        y_ret[:, :-1, :] = y[:, 1:, :]
    elif op == 'right' and y.size(1) > 1:
        A[:, 1:, :] = align[:, :-1, :]
        y_ret[:, 1:, :] = y[:, :-1, :]
    else:
        pass
    x_align = torch.bmm(x, A)
    align_cnt = torch.sum(align, dim=1)
    align_cnt[align_cnt == 0] = 1 # prevent devide by zero
    x_align = x_align / align_cnt[:,None,:] # normalize by number of alignemnt
    x_align = x_align.permute(0, 2, 1)  # [n, T2, d]
    return x_align, y_ret


#################################################################################
# Model class

class TakeFirst(nn.Module):
    def forward(self, x):
        return x[0]

def add_gap_tokens(x, emb, vocab, D, new_tensor):
    """ Add the <s> tokens after each token and the start of the sentence
        x: [n, T, d]
        Return x_grap: [n, 2T+1, d]
    """
    n, T, d = x.size()
    gap = [vocab.share['<s>']] * n  # [n, d]
    #print('gap', gap)
    x_gap = emb(Variable(new_tensor(gap)))
    #print('x_gap', x_gap.size(), x_gap)
    x_gap = x_gap.repeat(1, D)    # [n, d]
    x_new = [x_gap]
    for xi in x.split(split_size=1, dim=1):
        x_new.append(xi.squeeze(1))
        x_new.append(emb(Variable(new_tensor(gap))).repeat(1, D))
    x_new = torch.stack(x_new, dim=1)
    assert x_new.size(1) == 2 * T + 1, 'x_new.size(1)={}!={}'.format(x_new.size(1), 2 * T + 1)
    return x_new

def make_layers(cfg, input_size):
    layers = []
    for v in cfg:
        if v[0] == 'FF':
            layers += [nn.Linear(input_size, v[1], bias=v[2])]
            input_size = v[1]
        elif v[0] == 'GRU':
            layers += [nn.GRU(input_size, v[1], num_layers=v[2], bidirectional=v[3], batch_first=True), TakeFirst()]
            input_size = v[1] * 2
        elif v[0] == 'GRU_V2':
            layers += [nn.GRU(input_size, v[1], num_layers=v[2], bidirectional=v[3], batch_first=True), TakeFirst()]
            input_size = v[1] * 2
        elif v[0] == 'ReLU':
            layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

class Base(nn.Module):
    def __init__(self, args, vocab):
        super(Base, self).__init__()
        self.args = args
        self.vocab = vocab

    def uniform_init(self, v):
        print('uniformly initialize parameters [-%f, %f]' % (v, v))
        for p in self.parameters():
            p.data.uniform_(-v, v)

    def save(self, path):
        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

class Baseline(Base):
    def __init__(self, args, vocab, dropoute=.1, dropouti=.5):
        Base.__init__(self, args, vocab)
        self.dropoute = dropoute
        self.dropouti = dropouti

        # Load embeddings (torch.FloatTensor) from a file
        self.src_emb = nn.Embedding(len(vocab.src), args.embed_size, padding_idx=vocab.src['<pad>'])
        self.trg_emb = nn.Embedding(len(vocab.trg), args.embed_size, padding_idx=vocab.trg['<pad>'])
        if args.src_embed_file and args.trg_embed_file:
            src_embed = torch.from_numpy(load_embedding(vocab.src, args.src_embed_file, args.embed_size))
            trg_embed = torch.from_numpy(load_embedding(vocab.trg, args.trg_embed_file, args.embed_size))
            self.src_emb.weight.data.copy_(src_embed)
            self.trg_emb.weight.data.copy_(trg_embed)

        self.cfg = [('FF', 400, True), ('ReLU'), ('FF', 400, True), ('ReLU'), ('GRU', 200, 1, True),
                    ('FF', 200, True), ('ReLU'), ('FF', 200, True), ('ReLU'), ('GRU', 100, 1, True),
                    ('FF', 100, True), ('ReLU'), ('FF', 50, True), ('ReLU'), ('FF', 2, True)]

        #self.cfg = [('FF', 400, True), ('ReLU'), ('FF', 400, True), ('ReLU'), ('GRU_V2', 200, 1, True),
        #            ('FF', 200, True), ('ReLU'), ('FF', 200, True), ('ReLU'), ('GRU_V2', 100, 1, True),
        #            ('FF', 100, True), ('ReLU'), ('FF', 50, True), ('ReLU'), ('FF', 2, True)]
        self.feature = make_layers(self.cfg, args.embed_size * 6)

    def forward(self, src, hyp, align):
        """
        src: (n, T1)
        hyp: (n, T2)
        align: (n, T1, T2)
        """
        x = self.src_emb(src)   # [n, T1, d]
        y = self.trg_emb(hyp)   # [n, T2, d]

        x = x.permute(0, 2, 1)              # [n, d, T1]
        x_align_l, y_l = get_align_vector(x, y, align, op='left')
        x_align_m, y_m = get_align_vector(x, y, align, op='mid')
        x_align_r, y_r = get_align_vector(x, y, align, op='right')
        xy = torch.cat([x_align_l, x_align_m, x_align_r, y_l, y_m, y_r], dim=2)  # [n, T2, 6d]

        xy = self.feature(xy)    # [n, T2, 2]
        return xy

class CrosslingualBase(Base):
    def __init__(self, args, vocab):
        Base.__init__(self, args, vocab)
        self.emb = nn.Embedding(len(vocab.share), args.embed_size, padding_idx=vocab.share['<pad>'])
        if args.embed_file is not None:
            embed = torch.from_numpy(load_embedding(vocab.share, args.embed_file, args.embed_size))
            self.emb.weight.data.copy_(embed)

        #self.pos_emb = nn.Embedding(len(vocab.pos), args.pos_size)

        self.cfg = [('FF', 400, True), ('ReLU'), ('FF', 400, True), ('ReLU'), ('GRU', 200, 1, True),
                    ('FF', 200, True), ('ReLU'), ('FF', 200, True), ('ReLU'), ('GRU', 100, 1, True),
                    ('FF', 100, True), ('ReLU'), ('FF', 50, True), ('ReLU')]

        self.feature = make_layers(self.cfg, args.embed_size * 6)
        self.output = nn.Linear(50 + args.extra_feat_size, 2, bias=True)

    def forward(self, src, hyp, align, feat):
        """
        src: (n, T1)
        hyp: (n, T2)
        align: (n, T1, T2)
        feat: (n, T2, extra_feat_size)
        """
        x = self.emb(src)   # [n, T1, d]
        y = self.emb(hyp)   # [n, T2, d]

        x = x.permute(0, 2, 1)              # [n, d, T1]
        x_align_l, y_l = get_align_vector(x, y, align, op='left')
        x_align_m, y_m = get_align_vector(x, y, align, op='mid')
        x_align_r, y_r = get_align_vector(x, y, align, op='right')
        xy = torch.cat([x_align_l, x_align_m, x_align_r, y_l, y_m, y_r], dim=2)  # [n, T2, 6d]

        # add gap tokens or not
        if self.args.add_gap:
            xy = add_gap_tokens(xy, self.emb, self.vocab, 6, src.data.new)

        xy = self.feature(xy)    # [n, T2, D]
        if self.args.extra_feat_size > 0 and self.args.extra_feat_size == feat.size(-1):
            xy = torch.cat([xy, feat], dim=2)
        xy = self.output(xy)     # [n, T2, 2]
        return xy


class CrosslingualConv(Base):
    def __init__(self, args, vocab):
        Base.__init__(self, args, vocab)
        self.emb = nn.Embedding(len(vocab.share), args.embed_size, padding_idx=vocab.share['<pad>'])
        if args.embed_file is not None:
            embed = torch.from_numpy(load_embedding(vocab.share, args.embed_file, args.embed_size))
            self.emb.weight.data.copy_(embed)

        # kernel_size = (1, 3, 5, 7)
        self.kernel_size_list = [1, 3, 5, 7]
        for kernel_size in self.kernel_size_list:
            conv = torch.nn.Conv1d(args.embed_size * 6, args.conv_size,
                                   kernel_size, stride=1, dilation=1,
                                   padding=(kernel_size-1)//2,)
            setattr(self, 'fconv_{}'.format(kernel_size), conv)

        self.cfg = [('FF', 400, True), ('ReLU'), ('FF', 400, True), ('ReLU'), ('GRU', 200, 1, True),
                    ('FF', 200, True), ('ReLU'), ('FF', 200, True), ('ReLU'), ('GRU', 100, 1, True),
                    ('FF', 100, True), ('ReLU'), ('FF', 50, True), ('ReLU')]

        self.feature = make_layers(self.cfg, len(self.kernel_size_list) * args.conv_size)
        self.output = nn.Linear(50 + args.extra_feat_size, 2, bias=True)

    def get_conv(self, kernel_size):
        return getattr(self, 'fconv_{}'.format(kernel_size))

    def forward(self, src, hyp, align, feat=None):
        """
        src: (n, T1)
        hyp: (n, T2)
        align: (n, T1, T2)
        """
        x = self.emb(src)   # [n, T1, d]
        y = self.emb(hyp)   # [n, T2, d]

        x = x.permute(0, 2, 1)              # [n, d, T1]
        x_align_l, y_l = get_align_vector(x, y, align, op='left')
        x_align_m, y_m = get_align_vector(x, y, align, op='mid')
        x_align_r, y_r = get_align_vector(x, y, align, op='right')
        xy = torch.cat([x_align_l, x_align_m, x_align_r, y_l, y_m, y_r], dim=2)  # [n, T2, 6d]

        # add gap tokens or not
        #if self.args.add_gap:
        #    xy = add_gap_tokens(xy, self.emb, self.vocab, 6, src.data.new)

        # Conv1D on [n, 6d, T2] along T2 dimension
        xy = xy.permute(0, 2, 1)
        conv_out = []
        for kernel_size in self.kernel_size_list:
            xy_conv = F.relu(self.get_conv(kernel_size)(xy))
            conv_out.append(xy_conv)
        xy_conv = torch.cat(conv_out, 1).permute(0, 2, 1)   # [n, T2, len(kernel_size_list) * conv_size]

        xy = self.feature(xy_conv)    # [n, T2, D]
        if self.args.extra_feat_size > 0 and self.args.extra_feat_size == feat.size(-1):
            xy = torch.cat([xy, feat], dim=2)
        xy = self.output(xy)     # [n, T2, 2]
        return xy


