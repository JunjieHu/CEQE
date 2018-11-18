from __future__ import print_function
import argparse
from collections import Counter
from itertools import chain

import torch

from util import read_bitext, read_corpus


class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.id2word = {self.word2id[w]:w for w in self.word2id}
        # self.id2word = {v: k for k, v in self.word2id.iteritems()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    @staticmethod
    def from_corpus(corpus, size, remove_singleton=True):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        non_singletons = [w for w in word_freq if word_freq[w] > 1]
        print('number of word types: %d, number of word types w/ frequency > 1: %d' % (len(word_freq),
                                                                                       len(non_singletons)))

        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)[:size]

        for word in top_k_words:
            if len(vocab_entry) < size:
                if not (word_freq[word] == 1 and remove_singleton):
                    vocab_entry.add(word)

        return vocab_entry

    @staticmethod
    def from_bilingual_corpus(src_corpus, trg_corpus, src_size, trg_size, remove_singleton=True):
        src_vocab_entry = VocabEntry()
        trg_vocab_entry = VocabEntry()
        vocab_entry = VocabEntry()

        src_word_freq = Counter(chain(*src_corpus))
        trg_word_freq = Counter(chain(*trg_corpus))

        non_src_singletons = [w for w in src_word_freq if src_word_freq[w] > 1]
        non_trg_singletons = [w for w in trg_word_freq if trg_word_freq[w] > 1]
        print('SRC: no. of word types: %d, no. of word types w/ frequency > 1: %d' % (len(src_word_freq), len(non_src_singletons)))
        print('TRG: no. of word types: %d, no. of word types w/ frequency > 1: %d' % (len(trg_word_freq), len(non_trg_singletons)))

        top_src_words = sorted(src_word_freq.keys(), reverse=True, key=src_word_freq.get)[:src_size]
        top_trg_words = sorted(trg_word_freq.keys(), reverse=True, key=trg_word_freq.get)[:trg_size*2]

        for word in top_src_words:
            if len(vocab_entry) < src_size:
                if not (src_word_freq[word] == 1 and remove_singleton):
                    vocab_entry.add(word)
                    src_vocab_entry.add(word)
        print('generate %d / %d source words' % (len(src_vocab_entry), len(src_word_freq)))

        for word in top_trg_words:
            # if word in src_vocab_entry:               
            #     continue
            if len(trg_vocab_entry) < trg_size:
                if not (trg_word_freq[word] == 1 and remove_singleton):
                    vocab_entry.add(word)
                    trg_vocab_entry.add(word)
        print('generate %d / %d target words' % (len(trg_vocab_entry), len(trg_word_freq)))
        print('generate %d / %d shared words' % (len(vocab_entry), len(src_word_freq) + len(trg_word_freq)))
        return vocab_entry, src_vocab_entry, trg_vocab_entry

    @staticmethod
    def from_feature_corpus(feature_file):
        pos_set = set()
        src_pos, trg_pos = set(), set()
        vocab_entry = VocabEntry()
        vocab_entry.word2id = {'<unk>': 0}
        # 20, 21
        for line in open(feature_file, 'r'):
            items = line.strip().split('\t')
            if len(items) == 1 and items[0] == '':
                continue
            #print('items', len(items), items)
            vocab_entry.add(items[20])
            vocab_entry.add(items[21])
            src_pos.add(items[20])
            trg_pos.add(items[21])

        print('generate %d POS tags, source POS %d, target POS %d' % (len(vocab_entry), len(src_pos), len(trg_pos)))
        print('src - trg', src_pos - trg_pos)
        print('trg - src', trg_pos - src_pos)
        print('src intersection trg', src_pos.intersection(trg_pos))
        #print('src pos', src_pos)
        #print('trg pos', trg_pos)
        return vocab_entry


class Vocab(object):
    def __init__(self, src_sents, trg_sents, src_vocab_size, trg_vocab_size, remove_singleton=False, share_vocab=False, pos_file=None):
        assert len(src_sents) == len(trg_sents)

        if share_vocab:
            print('initialize share vocabulary ..')
            self.share, self.src, self.trg = VocabEntry.from_bilingual_corpus(src_sents, trg_sents, src_vocab_size, trg_vocab_size, remove_singleton=remove_singleton)
        else:
            print('initialize source vocabulary ..')
            self.src = VocabEntry.from_corpus(src_sents, src_vocab_size, remove_singleton=remove_singleton)

            print('initialize target vocabulary ..')
            self.trg = VocabEntry.from_corpus(trg_sents, trg_vocab_size, remove_singleton=remove_singleton)

        #if pos_file is not None:
        print('pos_file', pos_file)
        self.add_pos(pos_file)

    def __repr__(self):
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.trg))

    def add_pos(self, feature_file):
        self.pos = VocabEntry.from_feature_corpus(feature_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_vocab_size', default=50000, type=int, help='source vocabulary size')
    parser.add_argument('--trg_vocab_size', default=50000, type=int, help='target vocabulary size')
    parser.add_argument('--include_singleton', action='store_true', default=False, help='whether to include singleton'
                                                                                        'in the vocabulary (default=False)')

    parser.add_argument('--train_bitext', type=str, default='', help='file of parallel sentences')
    parser.add_argument('--train_src', type=str, help='path to the source side of the training sentences')
    parser.add_argument('--train_trg', type=str, help='path to the target side of the training sentences')
    parser.add_argument('--train_feature', type=str, help='path to the training feature of the training sentences')
    parser.add_argument('--output', default='vocab.bin', type=str, help='output vocabulary file')
    parser.add_argument('--share_vocab', action='store_true', default=False)
    parser.add_argument('--lowercase', action='store_true', default=False)

    args = parser.parse_args()

    print('read in parallel sentences: %s' % args.train_bitext)
    if args.train_bitext:
        src_sents, trg_sents = read_bitext(args.train_bitext)
    else:
        src_sents = read_corpus(args.train_src, source='src', lowercase=args.lowercase)
        trg_sents = read_corpus(args.train_trg, source='src', lowercase=args.lowercase)

    vocab = Vocab(src_sents, trg_sents, args.src_vocab_size, args.trg_vocab_size, remove_singleton=not args.include_singleton, share_vocab=args.share_vocab, pos_file=args.train_feature)
    print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.trg)))

    torch.save(vocab, args.output)
    print('vocabulary saved to %s' % args.output)


