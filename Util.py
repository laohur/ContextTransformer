"处理文本"
import argparse
import torch
import transformer.Constants as Constants
import json
import math
from time import time
import os
import numpy as np
import sys
import random

# device = "cuda:0" if torch.cuda.is_available() else "cpu"


def char_type(c):  # 判断字符类型
    # https://gist.github.com/shingchi/64c04e0dd2cbbfbc1350
    if ord(c) <= 0x007f:  # ascii
        if ord(c) >= 0x0030 and ord(c) <= 0x0039:
            return "number"
        if ord(c) >= 0x0041 and ord(c) <= 0x005a:
            return "latin"
        if ord(c) >= 0x0061 and ord(c) <= 0x007a:
            return "latin"
        return "ascii_symble"
    if ord(c) >= 0x4E00 and ord(c) <= 0x9fff:
        return "han"  # 标准CJK文字
    if ord(c) >= 0xFF00 and ord(c) <= 0xFFEF:
        return "han_symble"  # 全角ASCII、全角中英文标点、半宽片假名、半宽平假名、半宽韩文字母：FF00-FFEF
    if ord(c) >= 0x3000 and ord(c) <= 0x303F:
        return "han_symble"  # CJK标点符号：3000-303F
    return "other"


def split_lans(line):
    last_latin = None
    grams = []
    for gram in line:  # 还是字符串形式
        if char_type(gram) == "latin":
            if last_latin == None or last_latin == False:
                grams.append(gram)
            else:
                grams[-1] += gram
            last_latin = True
        else:
            grams.append(gram)
            last_latin = False
    return grams


def merge_gram(line):
    last_type = None
    tokens = []
    for gram in line:
        if char_type(gram) == "latin":
            if last_type == "latin":
                tokens[-1] += gram
            else:
                tokens.append(gram)
        elif char_type(gram) == "number":
            if last_type == "number":
                tokens[-1] += gram
            else:
                tokens.append(gram)
        else:
            if gram not in [None, '', ' ']:
                tokens.append(gram)
        last_type = char_type(gram)
    return tokens


def read_file(path, keep_case=False, begin=0, end=-1):
    if end < 0:
        end = sys.maxsize
    if "vocab" in path:
        print("vocab", path)
    t0 = time()
    print("load_file正在读取", os.path.abspath(path), begin, "->", end)
    doc = open(path, "r", encoding="utf-8").read().splitlines()[begin:end]
    for i in range(len(doc)):
        if not keep_case:
            doc[i] = doc[i].lower()
        if i % 100000 == 0:
            print("进展", i * 100.0 / len(doc), "第", i, "行", doc[i])
    print('[Info] ""文件{}中获取{}句子'.format(path, len(doc)), "耗时", time() - t0)

    return doc


def load_file(inst_file, max_sent_len, keep_case=False, begin=0, end=-1):
    if end < 0:
        end = math.inf
    if "vocab" in inst_file:
        print("vocab", inst_file)
    ''' 由文本生成词典和序列 '''
    word_insts = []
    trimmed_sent_count = 0
    t0 = time()
    print("load_file正在读取", os.path.abspath(inst_file), begin, "->", end)
    doc = open(inst_file, "r", encoding="utf-8").read().splitlines()
    for i in range(len(doc)):
        if i < begin:
            continue
        if i > end:
            break
        sent = doc[i]
        if not keep_case:
            sent = sent.lower()
        words = sent.split()
        if len(words) > max_sent_len:
            # if len(words)>max_sent_len*9:
            # print("过长被截断",len(words),words)
            trimmed_sent_count += 1
        # 左截断，左右各一半更好
        word_inst = words[:max_sent_len]
        line = [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]] if word_inst else [None]
        word_insts += line
        if i % 1000000 == 0:
            print("进展", i * 100.0 / len(doc), "第", i, "行", line)
    print('[Info] ""文件{}中获取{}句子'.format(inst_file, len(word_insts)))

    if trimmed_sent_count > 0:
        print('[Warning] {}个句子被截断至最大长度{}.'.format(trimmed_sent_count, max_sent_len))

    return word_insts


def build_vocab_idx(word_insts, min_word_count, max_words=100000):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] 原始词库 =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1
    word_count = dict(sorted(word_count.items(), key=lambda kv: kv[1], reverse=True))

    print("词频", word_count)
    ignored_word_count = 0
    for word, count in word_count.items():
        if len(word2idx) >= max_words:
            print("[!]词典已满", len(word2idx))
            break
        if word not in word2idx:
            if count >= min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    frequency = {}
    for word, idx in word2idx.items():
        if idx < 4:
            frequency[idx] = -1
        elif word in word_count:
            frequency[idx] = math.sqrt(word_count[word])

    print('[Info] 频繁字典大小 = {},'.format(len(word2idx)), '最低频数 = {}'.format(min_word_count))
    print("[Info] 忽略罕词数 = {}".format(ignored_word_count), "爆表词汇数", len(full_vocab) - max_words)
    return word2idx, frequency


def convert_w2id_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]


def line2idx(line, word2idx):
    idx = []
    for token in line.split(" "):
        id = word2idx.get(token, Constants.UNK)
        if id >= 4:  # 是否因为去除unk变差 4
            idx.append(id)
    return idx


def digitalize(src, tgt, ctx, max_sent_len, word2idx, index2freq, topk):
    t = 100000
    trimmed_src_count, trimmed_tgt_count, trimmed_ctx_count = 0, 0, 0
    for i in range(len(src)):
        if i % t == 0:
            print(i, "条src[i]，进度", i * 1.0 / len(src), src[i], "-->")
            print("\t ctx[i]", ctx[i])
        src[i] = line2idx(src[i], word2idx)
        ctx[i] = line2idx(ctx[i], word2idx)
        if tgt != None:
            if i % t == 0:
                print("\t tgt[i] ", "-->", tgt[i])
            tgt[i] = line2idx(tgt[i], word2idx)
            if len(tgt[i]) > max_sent_len:
                trimmed_tgt_count += 1
        # tgt[i] = [Constants.BOS_WORD] + line2idx(tgt[i], word2idx) + [Constants.EOS_WORD]
        if index2freq != None:
            if topk == 0 or random.random() < 0.3:
                tops = np.random.randint(low=4, high=len(word2idx) - 1, size=topk).tolist()
            else:
                tops = top_words(tgt[i], index2freq)
                tops = tops[:topk]
            src[i] += tops  # 从末尾加，长句多独特，短句更类似。
        if len(src[i]) > max_sent_len:
            trimmed_src_count += 1
        if len(ctx[i]) > max_sent_len:
            trimmed_ctx_count += 1
        src[i] = [Constants.BOS] + src[i][:max_sent_len] + [Constants.EOS]
        ctx[i] = [Constants.BOS] + ctx[i][:max_sent_len] + [Constants.EOS]
        if i % t == 0:
            print("\t src[i]---->", src[i])
            print("\t ctx[i]---->", ctx[i])
        if tgt != None:
            tgt[i] = [Constants.BOS] + tgt[i][:max_sent_len] + [Constants.EOS]
            if i % t == 0:
                print("\t tgt[i] ", "-->", tgt[i])
    print(trimmed_src_count, "条问题", trimmed_tgt_count, "条回答", trimmed_ctx_count, "条背景被截断至", max_sent_len)

    return src, ctx, tgt


def top_words(seq, index2freq):  # [0,1,2
    dt = {}
    for id in seq:
        # id -= 4  # [PAD,四个标记符]
        if id < 4:
            continue
        if id not in dt:
            dt[id] = 1
        else:
            dt[id] += 1

    for k in dt:
        dt[k] = dt[k] * 1.0 / index2freq[k]

    # re = {}  # 4起
    # for k, v in dt.items():
    #     if k >= len(index2freq):
    #         print(seq, len(index2freq))
    #         input("[error] top_words 列表溢出见上 ")
    #     re[k] = v * 1.0 / index2freq[k]  # 内外比值
    dt = sorted(dt.items(), key=lambda kv: kv[1], reverse=True)
    dt = dict(dt)
    return list(dt)


def add_keywords(src, tgt, word2index, topk, frequency):  # [2,9,67,4,3,0,0,0]
    for i in range(len(src)):
        tops = np.random.randint(low=4, high=len(word2index) - 1, size=topk).tolist()

        if tgt == None:
            tops = top_words(tgt[i], frequency)
            # tops = random.sample(tops, min(self.topk + 3, len(tops)))
            tops = tops[:topk]
            for idx in tops:
                if idx < 4:
                    print(tops)
                    input("topk <4", word2index[idx])

        src[i][1:1] = tops
    return src
