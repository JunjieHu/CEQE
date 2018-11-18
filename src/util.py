from collections import defaultdict, Counter
import numpy as np
import operator
import random
import argparse
import scipy
from scipy.stats import spearmanr, kendalltau
import math

def save_submission(file_path, prediction, data, method, prediction_type):
    """ prediction: [n, num_tags] 
        data: [n, num_tags]
    """
    with open(file_path, 'w') as f:
        for sid, (sent, tag) in enumerate(list(zip(data, prediction))):
            assert len(sent) == len(tag), 'len(sent)={} != len(tag)={}'.format(len(sent), len(tag))
            for i, (w, t) in enumerate(list(zip(sent, tag))):
                tlabel = 'OK' if t == 1 else 'BAD'
                f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(method, prediction_type, sid, i, w, tlabel))

def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]

# see /usr1/home/yuexinw/research/qe/data/WMT2018QE/features
# 28 original features
# def read_baseline_features(file_path):
#     data, sent = [], []
#     for line in open(file_path, 'r'):
#         feat = line.strip().split('\t')
#         if len(feat) == 1 and feat[0] == '':
#             data.append(sent)
#             sent = []
#             continue
#         remained_feature = list(map(float, feat[:3] + feat[9:20]))  # dim = 14
#         sent.append(remained_feature)
#     return data

def read_baseline_features(file_path):
    data, sent = [], []
    src_poses, src_pos = [], []
    trg_poses, trg_pos = [], []
    for line in open(file_path, 'r'):
        feat = line.strip().split('\t')
        if len(feat) == 1 and feat[0] == '':
            data.append(sent)
            sent = []
            src_poses.append(src_pos)
            src_pos = []
            trg_poses.append(trg_pos)
            trg_pos = []
            continue
        remained_feature = list(map(float, feat[9:13] + feat[15:18]))  # dim = 14
        # print('feat', line, feat)
        for n in (feat[13:15] + feat[18:20]):
            ngram = [0.0] * 6 
            ngram[int(n)] = 1.0
            remained_feature += ngram
        sent.append(remained_feature)


    return data

def read_corpus(file_path, source='src', lowercase=False):
    data = []
    for line in open(file_path, 'r'):
        sent = line.strip().split(' ')
        if lowercase:
            sent = [w.lower() for w in sent]
        # only append <s> and </s> to the target sentence
        if source == 'trg':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

def read_source(src_path, src_w2i, max_len=None, strict=False):
    src_sents = []
    for line in open(src_path, 'r'):
        src = [src_w2i[w] for w in line.strip().split()]
        if max_len is not None:
            src = src[0: max_len]
        src_sents.append(src)
        if strict:
            assert len(src_sents[-1]) <= max_len, str(len(src_sents[-1])) + ':' + line
    return src_sents

# def read_alignment(file_path):
#     alignments = []
#     for line in open(file_path):
#         pairs = [map(int, seg.split('-')) for seg in line.strip().split(' ')]
#         src_len = max(a[0] for a in pairs)
#         align = [[] for _ in range(src_len)]
#         for a in pairs:
#             align[a[0]].append(a[1])
#         alignments.append(align)
#     return alignments

def read_tags(file_path, skip_gap=False):
    tags = []
    for line in open(file_path):
        tag = line.strip().split(' ')
        tag = [1 if t == 'OK' else 0 for t in tag]
        if skip_gap:
            tag = tag[1:len(tag):2]
        tags.append(tag)
    return tags

def read_alignment_matrix(file_path, src_sents, ref_sents):
    alignments = []
    for line, src, ref in zip(open(file_path), src_sents, ref_sents):
        pairs = [[int(a) for a in seg.split('-')] for seg in line.strip().split(' ')]
        max_src_idx = max(a[0] for a in pairs)
        max_ref_idx = max(a[1] for a in pairs)
        assert max_src_idx < len(src), 'max src idx={} !< len(src)={}'.format(max_src_idx, len(src))
        assert max_ref_idx < len(ref), 'max ref idx={} !< len(ref)={}'.format(max_ref_idx, len(ref))
        matrix = [[0 for _ in range(len(ref))] for _ in range(len(src))]  # [T1, T2]
        for a in pairs:
            matrix[a[0]][a[1]] = 1
        alignments.append(matrix)
    return alignments # [n, T1, T2]

def read_alignment(file_path, src_sents, ref_sents, reverse=False, bos_offset=0):
    alignments, aligned_sents = [], []
    for line, src, ref in zip(open(file_path), src_sents, ref_sents):
        pairs = [[int(a) for a in seg.split('-')] for seg in line.strip().split(' ')]
        pairs = [(a[0] + bos_offset, a[1]) for a in pairs]
        max_src_idx = max(a[0] for a in pairs)
        max_ref_idx = max(a[1] for a in pairs)

        assert max_src_idx < len(src), 'max src idx={} !< len(src)={}'.format(max_src_idx, len(src))
        assert max_ref_idx < len(ref), 'max ref idx={} !< len(ref)={}'.format(max_ref_idx, len(ref))
        if reverse:
            # each target word aligns to n source words (n>=1)
            #max_len = max(a[1] for a in pairs) + 1
            max_len = len(ref)
            assert max_len == len(ref), 'max_len={} != len(ref)={}'.format(max_len, len(ref))
        else:
            # each source word aligns to m target words (m>=1)
            # max_len = max(a[0] for a in pairs) + 1
            max_len = len(src)
            assert max_len == len(src), 'max_len={} != len(src)={}'.format(max_len, len(src))

        align = [[] for _ in range(max_len)]
        aligned_sent = [[] for _ in range(max_len)]
        for a in pairs:
            if reverse:
                align[a[1]].append(a[0])
                aligned_sent[a[1]].append(src[a[0]])
            else:
                align[a[0]].append(a[1])
                aligned_sent[a[0]].append(ref[a[1]])
        alignments.append(align)
        aligned_sents.append(aligned_sent)
    return alignments, aligned_sents  # [n, T2, k]

def pad_sent(sent, PAD, max_len=None):
    new_sent, mask = [], []
    if max_len is None:
        max_len = max(len(x) for x in sent)
    for sid, s in enumerate(sent):
        #print('len(s) = ', len(s))
        mask.append([1] * len(s) + [0] * (max_len - len(s)))
        new_sent.append(s + [PAD] * (max_len - len(s)))
        #print('len', len(new_sent[sid]), len(mask[sid]))
    return new_sent, mask, max_len

def read_bitext(file_path, add_tag_to_trg=False):
    """ Read parallel text with the format: 'src ||| trg' """
    src_sents, trg_sents = [], []
    for line in open(file_path):
        src_trg = line.strip().split('|||')
        src = src_trg[0].strip().split(' ')
        if add_tag_to_trg:
            trg = ['<s>'] + src_trg[1].strip().split(' ') + ['</s>']
        else:
            trg = src_trg[1].strip().split(' ')
        src_sents.append(src)
        trg_sents.append(trg)
    return src_sents, trg_sents

def read_nbest(file_path):
    """ Read nbest list with format: sid ||| sent ||| features ||| stat score """
    result = []
    for line in open(file_path, 'r'):
        nbests = line.strip().split(' ||| ')
        assert len(nbests) == 4
        sid = int(nbests[0])
        sent = nbests[1].strip().split(' ')
        feat = [float(f.split('=')[1]) for f in nbests[2].strip().split(' ')]
        score = [float(s) for s in nbests[3].strip().split(' ')]
        result.append((sid, sent, feat, score))
    return result

def compute_grad_norm(model, norm_type=2, verbose=False):
    # parameters = model.parameters()
    # parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        p_norm = p.grad.data.norm(norm_type) ** (norm_type)
        if verbose:
            print('Name', name, ', Grad=', p_norm)
        total_norm += p_norm
    return total_norm ** (1. / norm_type)


def compute_param_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        p_norm = p.data.norm(norm_type) ** (norm_type)
        total_norm += p_norm
    return total_norm ** (1. / norm_type)


def plot_xy(xy, x, y, bleu, bleu_pred, src_i2w, trg_i2w, path, color=False):
    for i in range(xy.shape[0]):
        #fig = plt.figure(i+1)
        #ax = fig.add_subplot(111)
        plt.figure(i+1)
        #print(y[i])
        #print(max(i2w.keys()))
        plt.xlabel(' '.join([trg_i2w[j] for j in y[i]]).encode('utf-8'))
        plt.ylabel(' '.join([src_i2w[j] for j in x[i]]).encode('utf-8'))
        plt.title('bleu=%.3f, bleu_pred=%.3f, len(src)=%d, len(trg)=%d' % (bleu[i], bleu_pred[i], len(x[i]), len(y[i])))
        #print(type(xy[i]), xy.shape)
        plt.imshow(xy[i])
        if not color:
            plt.gray()
        plt.savefig(path + '{}.color{}.png'.format(i, color))


def batch_slice(data, batch_size, sort=True):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    if sort:
        data = sorted(data, key=lambda x: len(x[0]), reverse=True)
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        src_sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        trg_sents = [data[i * batch_size + b][1] for b in range(cur_batch_size)]
        align_sents = [data[i * batch_size + b][2] for b in range(cur_batch_size)]
        tag_sents = [data[i * batch_size + b][3] for b in range(cur_batch_size)]
        features = [data[i * batch_size + b][4] for b in range(cur_batch_size)]
        trg_orig_sents = [data[i * batch_size + b][5] for b in range(cur_batch_size)]
        # if sort:
        #     src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
        #     src_sents = [src_sents[src_id] for src_id in src_ids]
        #     trg_sents = [trg_sents[src_id] for src_id in src_ids]

        yield src_sents, trg_sents, align_sents, tag_sents, features, trg_orig_sents

# def test_iter(data):
#     for pair in data: 


def data_iter(data, batch_size, shuffle=True, sort=True, is_test=False):
    """
    randomly permute data, then sort by target length, and partition into batches
    ensure that the length of target sentences in each batch is decreasing
    """
    if is_test:
        for src, trg, align, tag, feature, trg_orig in data:
            yield [src], [trg], [align], [tag], [feature], [trg_orig]
    else:
        buckets = defaultdict(list)
        for pair in data:
            trg_sent = pair[1]
            buckets[len(trg_sent)].append(pair)

        batched_data = []
        for trg_len in buckets:
            tuples = buckets[trg_len]
            #TODO: the following contain bug!!!
            #if shuffle: np.random.shuffle(tuples)
            batched_data.extend(list(batch_slice(tuples, batch_size, sort)))

        if shuffle:
            np.random.shuffle(batched_data)
        for batch in batched_data:
            yield batch

def compute_grad_norm(model, norm_type=2, verbose=False):
    # parameters = model.parameters()
    # parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        p_norm = p.grad.data.norm(norm_type) ** (norm_type)
        if verbose:
            print('Name', name, ', Grad=', p_norm)
        total_norm += p_norm
    return total_norm ** (1. / norm_type)


def compute_param_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        p_norm = p.data.norm(norm_type) ** (norm_type)
        total_norm += p_norm
    return total_norm ** (1. / norm_type)

def eval_metric(pred_np, label_np):
    """ measure the ranking between pred and label with size [n, m]
    """
    sp, kd = [], []
    for i in range(pred_np.shape[0]):
        sp.append(spearmanr(pred_np[i], label_np[i])[0])
        kd.append(kendalltau(pred_np[i], label_np[i])[0])
    sp_cnt = Counter(sp)
    kd_cnt = Counter(kd)
    # print('sp NaN:', sp_cnt['NaN'], len(sp), ', kd NaN:', kd_cnt['NaN'], len(kd))
    sp = [0 if math.isnan(x) else x for x in sp]
    kd = [0 if math.isnan(x) else x for x in kd]
    return np.mean(sp), np.mean(kd)


def get_statistic(nbest_path, src_path, trg_path, log_file):
    nbest = read_nbest(nbest_path)
    src_sents = read_corpus(src_path, source='src')
    trg_sents = read_corpus(trg_path, source='src')
    cnt = Counter()
    len_src = []
    len_hyp = []
    len_trg = []
    scores = []
    data = []
    sid_dict = {}
    for (sid, hyp, feat, score) in nbest:
        cnt[(len(src_sents[sid]), len(hyp))] += 1
        len_src.append(len(src_sents[sid]))
        len_trg.append(len(trg_sents[sid]))
        len_hyp.append(len(hyp))
        scores.append(score)

        # Get ranking list
        if sid not in sid_dict:
            sid_dict[sid] = len(data)
            data.append((src_sents[sid], [hyp_feat[1]], [score[0]]))
        else:
            idx = sid_dict[sid]
            assert src_sents[sid] == data[idx][0], 'src sent not match {} != {}'.format(src_sents[sid], data[idx][0])
            assert type(data[idx][1]) is list and type(data[idx][2]) is list
            data[idx][1].append(hyp_feat[1])    # negative log-likelihood
            data[idx][2].append(hyp_feat[0])    # BLEU

    scores_np = np.array(scores)
    with open(log_file, 'w') as f:
        f.write('avg len(src) = {}, avg len(hyp) = {}\n'.format(np.mean(len_src), np.mean(len_hyp), np.mean(len_trg)))
        f.write('max len(src) = {}, max len(hyp) = {}\n'.format(np.max(len_src), np.max(len_hyp), np.max(len_trg)))
        f.write('min len(src) = {}, min len(hyp) = {}\n'.format(np.min(len_src), np.min(len_hyp), np.min(len_trg)))
        f.write('Len src-hyp: \t count\n')
        score_mean = np.mean(scores_np, axis=0)
        f.write('mean of score: {}\n '.format(score_mean))
        f.write('mean baseline: {}\n '.format(np.mean(np.absolute(scores_np - score_mean), axis=0)))
        f.write('random baseline: {}\n '.format(np.mean(np.absolute(scores_np - np.random.rand(scores_np.shape[0], scores_np.shape[1])), axis=0)))
        for key, value in sorted(cnt.iteritems(), key=operator.itemgetter(1), reverse=True):
            f.write('{}-{}: \t {}\n'.format(key[0], key[1], value))

def get_stat(path, src_path, trg_path, log_file, max_len=60):
    src_sents = read_corpus(src_path, source='src')
    trg_sents = read_corpus(trg_path, source='src')
    len_src = [len(s) for s in src_sents]
    len_trg = [len(t) for t in trg_sents]
    len_hyp = []
    cnt = Counter()
    data = convert_nbest(path, src_sents, trg_sents, max_len)
    sp, kd = [], []
    scores = []
    for (src, hyp_sents, score, hyp_feats) in data:
        len_hyp.extend([len(h) for h in hyp_sents])
        for h in hyp_sents:
            cnt[(len(src), len(h))] +=1
        hyp_nll = [yy[1] for yy in hyp_feats]
        sp.append(spearmanr(hyp_nll, score)[0])
        kd.append(kendalltau(hyp_nll, score)[0])
        scores.append(score)
    sp_cnt = Counter(sp)
    kd_cnt = Counter(kd)
    # print('sp NaN:', sp_cnt['NaN'], len(sp), ', kd NaN:', kd_cnt['NaN'], len(kd))
    sp = [0 if math.isnan(x) else x for x in sp]
    kd = [0 if math.isnan(x) else x for x in kd]
    with open(log_file, 'w') as f:
        f.write('avg spearman coefficient: {}'.format(np.mean(sp)))
        f.write('avg kendalltau coefficient: {}'.format(np.mean(kd)))
        f.write('avg len(src) = {}, avg len(hyp) = {}\n'.format(np.mean(len_src), np.mean(len_hyp), np.mean(len_trg)))
        f.write('max len(src) = {}, max len(hyp) = {}\n'.format(np.max(len_src), np.max(len_hyp), np.max(len_trg)))
        f.write('min len(src) = {}, min len(hyp) = {}\n'.format(np.min(len_src), np.min(len_hyp), np.min(len_trg)))
        # f.write('Len src-hyp: \t count\n')

        scores = np.array(scores)
        score_mean = np.mean(scores)
        f.write('mean of score: {}\n '.format(score_mean))
        f.write('mean baseline: {}\n '.format(np.mean(np.absolute(scores - score_mean))))
        f.write('random baseline: {}\n '.format(np.mean(np.absolute(scores - np.random.rand(scores.shape[0], scores.shape[1])))))
        for key, value in sorted(cnt.iteritems(), key=operator.itemgetter(1), reverse=True):
            f.write('{}-{}: \t {}\n'.format(key[0], key[1], value))

def convert_nbest(path, src_sents, trg_sents, max_len, w2i=None, score_idx=0, scaled=False):
    data = []
    # source sentence index in the data, map sid to idx in data
    sid_dict = {}
    for line in open(path, 'r'):
        items = line.strip().split(' ||| ')
        sid = int(items[0])
        if w2i is None:
            hyp_sent = items[1].strip().split(' ')
        else:
            hyp_sent = [w2i[x] for x in items[1].strip().split(' ')]

        hyp_feat = [float(f.split('=')[1]) for f in items[2].strip().split(' ')]
        score = float(items[3].split()[score_idx])
        if scaled:
            score *= len(trg_sents[sid])
        if sid not in sid_dict:
            sid_dict[sid] = len(data)
            data.append((src_sents[sid], [hyp_sent], [score], [hyp_feat]))
        else:
            idx = sid_dict[sid]
            assert src_sents[sid] == data[idx][0], 'src sent not match {} != {}'.format(src_sents[sid], data[idx][0])
            assert type(data[idx][1]) is list and type(data[idx][2]) is list and type(data[idx][3]) is list, 'type(hyps) or type(scores) is not list'
            assert type(data[idx][1][-1]) is list, 'the last hyp sentence is not a list'
            assert type(data[idx][3][-1]) is list, 'the last hyp feature is not a list'
            data[idx][1].append(hyp_sent)
            data[idx][2].append(score)
            data[idx][3].append(hyp_feat)
    # ensure every source sentence have 5 hypotheses
    max_num_hyp = max([len(x[1]) for x in data])
    for slist in data:
        if len(slist[1]) != max_num_hyp:
            UNK = 0 if w2i is None else w2i["<unk>"]
            slist[1].extend([[UNK]] * (max_num_hyp - len(slist[1])))
            slist[2].extend([0] * (max_num_hyp - len(slist[2])))
            slist[3].extend([[0] * len(hyp_feat)] * (max_num_hyp - len(slist[3])))
        assert len(slist[1]) == max_num_hyp and len(slist[2]) == max_num_hyp and len(slist[3]) == max_num_hyp
    return data


def subsample(nbest_file, sent_path, nbest_out, sent_out, sample_size, bucket_size=10):
    random.seed(0)
    buckets = [[] for _ in range(bucket_size)]
    for line in open(nbest_file, 'r'):
        nbests = line.split(' ||| ')
        assert len(nbests) == 4
        score = [float(s) for s in nbests[3].strip().split(' ')]
        bid = int((score[0] - 1e-10) * bucket_size)
        assert 0 <= bid <= bucket_size - 1, bid
        buckets[bid].append(line)
    len_list = [len(b) for b in buckets]
    min_num = np.min(len_list)
    print('min num of distribution', min_num)
    print(len_list)
    sent_dict = defaultdict(lambda: len(sent_dict))
    cnt = 0
    with open(nbest_out, 'w') as fo:
        for i in range(bucket_size):
            blist = buckets[i]
            random.shuffle(blist)
            for lid, line in enumerate(blist):
                if lid >= np.min(sample_size):
                    break
                nbests = line.strip().split(' ||| ')
                sid = int(nbests[0])
                fo.write(" ||| ".join([str(sent_dict[sid])] + nbests[1:]) + "\n")
            print('sample {} nbests from {}-th buckets'.format(lid, i))
            cnt += lid
    print('sample {}/{} nbest'.format(cnt, np.sum(len_list)))

    sents = [l for l in open(sent_path, 'r')]
    print('sample {}/{} bitext sentences'.format(len(sent_dict), len(sents)))
    with open(sent_out, 'w') as f:
        for k, v in sorted(sent_dict.iteritems(), key=lambda x: x[1]):
            f.write(sents[k])

def get_ngram_match(ref, m_hyps, max_len, ngrams=4):
    """
    ref: [T1], m_hyps: [m, T2]
    m_matches: [m, ngrams, max_len], m_mask: [m, ngrams, max_len]
    """
    m_matches, m_masks = [], []
    for hyp in m_hyps:
        matches, masks = [], []
        for n in range(1, 1 + ngrams):
            ref_ngrams = [ref[i:i+n] for i in range(len(ref)-n+1)]
            hyp_ngrams = [hyp[i:i+n] for i in range(len(hyp)-n+1)]
            match = [1 if word in ref_ngrams else 0 for word in hyp_ngrams]
            mask = [0] * len(hyp_ngrams)
            if len(hyp_ngrams) < max_len:
                match.extend([0] * (max_len - len(hyp_ngrams)))
                mask.extend([1] * (max_len - len(hyp_ngrams)))
            matches.append(match)
            masks.append(mask)
        m_matches.append(matches)
        m_masks.append(masks)
    return m_matches, m_masks # [m, K, T2]

def stack_ngram_match(batch_refs, batch_m_hyps, max_len=None, ngrams=4):
    """
    batch_refs: [n, T1], batch_m_hyps: [n, m, T2]
    """
    max_len_ = np.max([[len(h) for h in m_hyps] for m_hyps in batch_m_hyps])
    max_len = max(max_len, max_len_)
    matches, masks = [], []
    for ref, m_hyps in zip(batch_refs, batch_m_hyps):
        match, mask = get_ngram_match(ref, m_hyps, max_len, ngrams)
        matches.append(match)
        masks.append(mask)
    return matches, masks  # [n, m, K, T2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_file', type=str)
    parser.add_argument('--nbest_file', type=str)
    parser.add_argument('--src_file', type=str)
    parser.add_argument('--trg_file', type=str)
    args = parser.parse_args()
    get_stat(args.nbest_file, args.src_file, args.trg_file, args.save_file)
