from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from util import save_submission, read_corpus, read_tags, read_alignment_matrix, read_baseline_features, data_iter, word2id, compute_grad_norm, compute_param_norm
import sys, os
from sklearn.metrics import f1_score
from model import Baseline, CrosslingualBase, CrosslingualConv
import argparse
import time
from vocab import Vocab, VocabEntry
import numpy as np
import random
from random import shuffle
import torch.optim.lr_scheduler as lr_scheduler
import pickle
############################################################################
# Define hype-parameters
def init_config():
    print('Initialize config')
    parser = argparse.ArgumentParser()
    # Switch prediction task
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # Tunable parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--uniform_init', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--min_lr', type=float, default=1e-7)
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--conv_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    # Operations
    parser.add_argument('--model', default='Baseline', choices=['Baseline', 'CrosslingualBase', 'CrosslingualConv'])
    parser.add_argument('--share_vocab', action='store_true', default=False)
    # Loss function
    parser.add_argument('--loss', type=str, default='ListwiseLoss', choices=['PairwiseLoss', 'ListwiseLoss', 'L1Loss', 'MSELoss', 'SmoothL1Loss'])
    parser.add_argument('--phi', type=str, default='hinge', choices=['hinge', 'exp', 'log', 'adaptive_hinge', 'cost_hinge'])
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--hinge_margin', type=float, default=1.0)
    parser.add_argument('--base', type=float, default=2.0)
    parser.add_argument('--valid_metric', type=str, default='f1-multi', choices=['f1-multi', 'f1-bad', 'f1-good', 'loss'])
    parser.add_argument('--weight_decay', type=float, default=0)
    # Data processing
    parser.add_argument('--max_len', type=int, default=None)
    parser.add_argument('--random_shuffle', action='store_true', default=False)
    parser.add_argument('--sort', action='store_true', default=False)
    parser.add_argument('--add_gap', action='store_true', default=False)
    parser.add_argument('--skip_gap', action='store_true', default=False)
    parser.add_argument('--extra_feat_size', type=int, default=0)
    # File path
    parser.add_argument('--vocab', type=str, default=None)
    parser.add_argument('--embed_file', type=str, default=None)
    parser.add_argument('--src_embed_file', type=str, default=None)
    parser.add_argument('--trg_embed_file', type=str, default=None)
    parser.add_argument('--save_model', type=str, default='rank-model')
    parser.add_argument('--trn_dev_path', type=str)
    parser.add_argument('--baseline_feature_path', type=str)
    parser.add_argument('--tst_path', type=str)
    parser.add_argument('--suffix', type=str, default='src mt align tags pe ref src_tags')
    parser.add_argument('--prefix', type=str, default='train dev test')
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--save_submission', type=str, default=None)
    parser.add_argument('--submission_name', type=str, default='CrosslingualConv')
    parser.add_argument('--prediction_type', type=str, default='mt')
    # Log & evaluation
    parser.add_argument('--log_niter', type=int, default=50)
    parser.add_argument('--valid_niter', type=int, default=200)
    parser.add_argument('--save_model_after', type=int, default=1)
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--cuda', default=False, action='store_true')

    args = parser.parse_args()
    print('args')
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))
    random.seed(int(args.seed * 13 / 7))
    return args

def load_data(path, split, suffix, skip_gap, feature_path, is_test=False):
    print('load data from {}'.format(path))
    if path is None:
        return None
    slist = suffix.split(' ')   # src mt src-mt.alignments tags pe ref src_tags features
    src_sents = read_corpus(path + '/%s.%s' % (split, slist[0]))
    hyp_sents_orig = read_corpus(path + '/%s.%s' % (split, slist[1]), lowercase=False)
    hyp_sents = [[w.lower() for w in hyp] for hyp in hyp_sents_orig]
    align_sents = read_alignment_matrix(path + '/%s.%s' % (split, slist[2]), src_sents, hyp_sents)
 
    if is_test:
        tag_sents = [[1] * len(hyp) for hyp in hyp_sents]
    else:
        tag_sents = read_tags(path + '/%s.%s' % (split, slist[3]), skip_gap)
    baseline_sents = read_baseline_features(feature_path + '/%s.%s' %(split, slist[7]))
    #baseline_sents = [[0 for h in hyp] for hyp in hyp_sents]
    return list(zip(src_sents, hyp_sents, align_sents, tag_sents, baseline_sents, hyp_sents_orig))

def init_model(args):
    print('load vocab from' + args.vocab)
    vocab = torch.load(args.vocab)

    if args.share_vocab:
        if args.model == 'CrosslingualBase':
            print('using CrosslingualBase model')
            model = CrosslingualBase(args, vocab)
        elif args.model == 'CrosslingualConv':
            print('using CrosslingualConv model')
            model = CrosslingualConv(args, vocab)
        else:
            print('model not exits')
            exit(0)
    else:
        if args.model == 'Baseline':
            print('using Baseline model')
            model = Baseline(args, vocab)
        else:
            print('model not exits')
            exit(0)

    # initialize model
    if args.uniform_init:
        model.uniform_init(args.uniform_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay, amsgrad=True)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)

    eval_loss_func = train_loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.0, 1.0]))
    if args.cuda:
        model = model.cuda()
        eval_loss_func = eval_loss_func.cuda()
        train_loss_func = train_loss_func.cuda()
    return model, vocab, optimizer, eval_loss_func, train_loss_func

def pad_source(src, PAD, vocab=None):
    """
    src: [n, T1]
    """
    if vocab is not None:
        src = [[vocab[w] for w in s] for s in src]
        PAD = vocab[PAD]
    max_len = max(len(s) for s in src)
    for s in src:
        s += [PAD] * (max_len - len(s))
    return src, max_len

def pad_align(align):
    """ align: [n, T1, T2] """
    src_len = max(len(a) for a in align)
    trg_len = max(len(a[0]) for a in align)

    #assert len(align) == list(len(ai) for ai in align).count(trg_len), 'len(trg)={}'.format([len(ai) for ai in align])
    for a in align:
        a += [[0] * trg_len] * (src_len - len(a))
    return align, src_len, trg_len

def flatten_list(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

def eval(args, data, model, eval_loss_func, vocab_src, vocab_trg, prefix='test', save_to_file=False, is_test=False):
    # if data is None:
    #     return 0, 0, 0, 0
    model.eval()
    all_tags_pred, all_tags_true = [], []
    all_tags_prob = []
    all_words = []
    cum_loss = cum_examples = 0.0
    i = 0
    print('len of data set in eval', len(data))

    for (src, hyp, align, tag, feat, hyp_orig) in data_iter(data, batch_size=args.batch_size, shuffle=False, sort=False, is_test=is_test):
        src_pad, src_len = pad_source(src, '<pad>', vocab_src)
        hyp = word2id(hyp, vocab_trg)
        align_pad, max_src_len, max_trg_len = pad_align(align)
        assert src_len == max_src_len, 'src_len={} != max_src_len={}'.format(src_len, max_src_len)
        assert len(hyp[0]) == len(feat[0])

        src_v = Variable(torch.LongTensor(src_pad))      # [n, T1]
        hyp_v = Variable(torch.LongTensor(hyp))          # [n, T2]
        align_v = Variable(torch.FloatTensor(align_pad)) # [n, T1, T2]
        tags_v = Variable(torch.LongTensor(tag))         # [n, T2]
        feat_v = Variable(torch.FloatTensor(feat))       # [n, T2, extra_feat_size]

        if args.cuda:
            src_v = src_v.cuda()
            hyp_v = hyp_v.cuda()
            align_v = align_v.cuda()
            tags_v = tags_v.cuda()
            feat_v = feat_v.cuda()

        tags_pred = model(src_v, hyp_v, align_v, feat_v)
        loss = eval_loss_func(tags_pred.view(-1, 2), tags_v.view(-1))
        cum_loss += loss.item() * len(hyp) * len(hyp[0])
        cum_examples += len(hyp) * len(hyp[0])

        all_tags_prob.extend(tags_pred.cpu().data.numpy().tolist())    # [n, T2, 2]
        all_tags_pred.extend(torch.max(tags_pred, dim=-1)[1].cpu().data.numpy().tolist())  # [n, T2]
        all_tags_true.extend(tag)                                                          # [n, T2]
        # hyp_words = [[model.vocab.share.id2word[w] for w in sent] for sent in hyp_orig]
        all_words.extend(hyp_orig)
        # all_tags_pred.extend(torch.max(tags_pred, dim=-1)[1].cpu().data.numpy().flatten())
        # all_tags_true.extend(tags_v.cpu().data.numpy().flatten())
    f1_bad, f1_good = f1_score(flatten_list(all_tags_true), flatten_list(all_tags_pred), average=None, pos_label=None)
    # print("F1-BAD: ", f1_bad, "F1-OK: ", f1_good)
    # print("F1-score multiplied: ", f1_bad * f1_good)

    if save_to_file:
        submission = {'probability': all_tags_prob, 'prediction': all_tags_pred, 'target': all_tags_true, 'words': all_words}
        pickle.dump(submission, open(args.save_submission + prefix + '.pkl', 'wb'))
        save_submission(args.save_submission + prefix + '.txt', all_tags_pred, all_words, args.submission_name, args.prediction_type)

    model.train()
    return cum_loss / cum_examples, f1_bad * f1_good, f1_bad, f1_good


def train(args):
    model, vocab, optimizer, eval_loss_func, train_loss_func = init_model(args)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_decay, patience=1, min_lr=1e-6)
    if args.share_vocab:
        vocab_src = vocab_trg = vocab.share
    else:
        vocab_src = vocab.src
        vocab_trg = vocab.trg

    train_data = load_data(args.trn_dev_path, 'train', args.suffix, args.skip_gap, args.baseline_feature_path)
    dev_data = load_data(args.trn_dev_path, 'dev', args.suffix, args.skip_gap, args.baseline_feature_path)
    test_data = load_data(args.tst_path, 'test', args.suffix, args.skip_gap, args.baseline_feature_path, is_test=True)

    print('begin MLE training')
    train_time = start_time = time.time()
    epoch = train_iter = 0
    cum_train_loss = cum_train_example = train_loss = train_example = valid_num = 0.0
    cum_tags_pred, cum_tags_true, train_tags_pred, train_tags_true = [], [], [], []
    hist_valid_scores = []
    while True:
        epoch += 1
        for (src, hyp, align, tag, feat, hyp_orig) in data_iter(train_data, batch_size=args.batch_size, shuffle=True):
            train_iter += 1

            src_pad, src_len = pad_source(src, '<pad>', vocab_src)
            align_pad, max_src_len, max_trg_len = pad_align(align)
            hyp = word2id(hyp, vocab_trg)
            assert src_len == max_src_len, 'src_len={} != max_src_len={}'.format(src_len, max_src_len)
            assert len(hyp[0]) == len(feat[0])

            src_v = Variable(torch.LongTensor(src_pad))      # [n, T1]
            hyp_v = Variable(torch.LongTensor(hyp))           # [n, T2]
            align_v = Variable(torch.FloatTensor(align_pad))  # [n, T1, T2]
            tags_v = Variable(torch.LongTensor(tag))         # [n, T2]
            feat_v = Variable(torch.FloatTensor(feat))       # [n, T2, d]
            if args.cuda:
                src_v = src_v.cuda()
                hyp_v = hyp_v.cuda()
                align_v = align_v.cuda()
                tags_v = tags_v.cuda()
                feat_v = feat_v.cuda()
            #print('src, hyp, tags', src_v.size(), hyp_v.size(), tags_v.size(), align_v.size())
            tags_pred = model(src_v, hyp_v, align_v, feat_v)    # [n, T2, 2]
            #print('tags_pred', tags_pred.size())
            loss = train_loss_func(tags_pred.view(-1, 2), tags_v.view(-1))
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item() * len(hyp) * len(hyp[0])
            train_example += len(hyp) * len(hyp[0])
            cum_train_loss += loss.item() * len(hyp) * len(hyp[0])
            cum_train_example += len(hyp) * len(hyp[0])

            tags_pred_label = torch.max(tags_pred, dim=-1)[1]
            train_tags_pred.extend(tags_pred_label.cpu().data.numpy().flatten())
            train_tags_true.extend(tags_v.cpu().data.numpy().flatten())
            cum_tags_pred.extend(tags_pred_label.cpu().data.numpy().flatten())
            cum_tags_true.extend(tags_v.cpu().data.numpy().flatten())


            if train_iter % args.log_niter == 0:
                train_tags_true = [int(t) for t in train_tags_true]
                train_tags_pred = [int(t) for t in train_tags_pred]
                #print('true tags', type(train_tags_true), type(train_tags_true[0]), len(train_tags_true), train_tags_true)
                #print('pred tags', type(train_tags_pred), type(train_tags_pred[0]), len(train_tags_pred), train_tags_pred)
                output = f1_score(train_tags_true, train_tags_pred, average=None, pos_label=None)
                #print(output)
                f1_bad, f1_good = output
                gnorm = compute_grad_norm(model)
                pnorm = compute_param_norm(model)
                print("Epoch %r, iter %r: train loss=%.4f, train f1-multi=%.4f, train f1-bad=%.4f, train f1-good=%.4f," \
                      "grad_norm=%.4f, p_norm=%.4f, time=%.2fs" % (epoch, train_iter, train_loss / train_example, \
                        f1_bad * f1_good, f1_bad, f1_good, gnorm, pnorm, time.time() - start_time))
                train_loss = train_example= 0.0
                train_tags_pred, train_tags_true = [], []

            # Perform dev
            if train_iter % args.valid_niter == 0:
                valid_num += 1
                model.eval()
                print('Begin validation ...')
                dev_loss, dev_multi, dev_bad, dev_good = eval(args, dev_data, model, eval_loss_func, vocab_src, vocab_trg, prefix='dev.iter'+str(train_iter), save_to_file=True)
                tst_loss, tst_multi, tst_bad, tst_good = eval(args, test_data, model, eval_loss_func, vocab_src, vocab_trg, prefix='test.iter'+str(train_iter), save_to_file=True, is_test=True)
                trn_loss = cum_train_loss / cum_train_example
                trn_bad, trn_good = f1_score(cum_tags_true, cum_tags_pred, average=None, pos_label=None)
                trn_multi = trn_bad * trn_good
                # trn_sp = trn_kd = 0
                print("validation: epoch %d, iter %d, train loss=%.4f, lr=%e, " \
                      "dev loss=%.4f, test loss=%.4f, time=%.2fs" % (epoch, train_iter, trn_loss, optimizer.param_groups[0]['lr'],
                                                                     dev_loss, tst_loss, time.time() - start_time))
                print("   iter %d, train multi=%.4f, dev multi=%.4f, test multi=%.4f" % (train_iter, trn_multi, dev_multi, tst_multi))
                print("   iter %d, train bad=%.4f, dev bad=%.4f, test bad=%.4f" % (train_iter, trn_bad, dev_bad, tst_bad))
                print("   iter %d, train good=%.4f, dev good=%.4f, test good=%.4f" % (train_iter, trn_good, dev_good, tst_good))
                if args.valid_metric == 'f1-multi':
                    valid_metric = dev_multi
                elif args.valid_metric == 'f1-bad':
                    valid_metric = dev_bad
                elif args.valid_metric == 'f1-good':
                    valid_metric = dev_good
                else:
                    valid_metric = - dev_loss
                cum_train_loss = cum_train_example = 0.0
                cum_tags_pred, cum_tags_true = [], []
                model.train()

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                is_better_than_last = len(hist_valid_scores) == 0 or valid_metric > hist_valid_scores[-1]
                hist_valid_scores.append(valid_metric)

                if valid_num > args.save_model_after:
                    model_file = args.save_model + '.iter{}.bin'.format(train_iter)
                    model.save(model_file)
                    print('save model to [%s]' % model_file)

                #if (not is_better_than_last) and args.lr_decay:
                #    lr = max(optimizer.param_groups[0]['lr'] * args.lr_decay, args.min_lr)
                #    print('decay learning rate to %e' % lr)
                #    optimizer.param_groups[0]['lr'] = lr
                scheduler.step(dev_loss)

                if is_better:
                    patience = 0
                    best_model_iter = train_iter

                    if valid_num > args.save_model_after:
                        print('save current best model ... ')
                        model_file_abs_path = os.path.abspath(model_file)
                        symlin_file_abs_path = os.path.abspath(args.save_model + '.bin')
                        os.system('ln -sf %s %s' % (model_file_abs_path, symlin_file_abs_path))
                else:
                    patience += 1
                    print('hit patience %d' % patience)
                    if patience == args.patience:
                        print('early stop! the best model is from [%d], best valid score %f' % (best_model_iter, max(hist_valid_scores)))
                        exit(0)


if __name__ == '__main__':
    args = init_config()

    train(args)
