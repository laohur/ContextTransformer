"处理文本"
import argparse
import torch
import transformer.Constants as Constants
import json
import math
from time import time
import os
from Util import merge_gram,splitgrams
import sys


def tokenize(line):
    # return list(line)
    # return line.split(" ")
    # words = line.split()
    # line = ''.join(line.split(" "))
    # words = split_lans(line)
    # words = merge_gram(line)
    words=splitgrams(line)
    # line = ' '.join(words)
    # words = line.split(" ")
    re = []
    for word in words:
        # if len(word) > 9:  # "13653923571"空值
        #     continue
        # word=word[:9]  #初期不要干
        if word in [None, "", " "]:
            continue
        re.append(word)
    return re


def token_file(source, target, counter, begin=0, end=sys.maxsize, keep_case=False):
    if end < 0:
        end = sys.maxsize

    print("token_file分词", os.path.abspath(source), "将写入", os.path.abspath(target))
    if counter == None:
        print(source, "不计入词频")
    f1 = open(source, "r", encoding="utf-8")
    f2 = open(target, "w", encoding="utf-8")
    line = f1.readline()
    print("第一条", line)
    i = 0
    while (line):
        i += 1
        if i < begin:
            continue
        if i > end:
            break
        if not keep_case:
            line = line.lower()
        tokens = tokenize(line)

        # 统计词频,attr除外
        if counter != None:
            for token in tokens:
                if token in counter:
                    counter[token] += 1
                else:
                    counter[token] = 1

        tokens = " ".join(tokens)  # 自带"\n"
        f2.write(tokens)
        if i % 100000 == 0:
            print("第", i, "行", line, "--->", tokens)
            if counter != None:
                print("已统计词频", len(counter))
        line = f1.readline()
    f1.close()
    f2.close()
    print(source, "已全部分词至", target)


def build_vocab_idx(word_count, min_word_count, max_words=100000):
    ''' Trim vocab by number of occurence '''
    print('[Info] 原始词库 =', len(word_count))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    # print("词频", word_count)
    ignored_word_count = 0
    for word, count in word_count.items():
        if len(word2idx) >= max_words:
            print("[!]词典已满", len(word2idx))
            break
        if len(word) > 7:
            continue
        if word not in word2idx:
            if count >= min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    frequency = [-1]*len(word2idx)
    for word, idx in word2idx.items():
        if idx < 4:
            frequency[idx] = -1
        elif word in word_count:
            frequency[idx] = math.sqrt(word_count[word])

    print('[Info] 频繁字典大小 = {},'.format(len(word2idx)), '最低频数 = {}'.format(min_word_count))
    print("[Info] 忽略罕词数 = {}".format(ignored_word_count), "爆表词汇数", len(word_count) - max_words)
    return word2idx, frequency


def count_file(mydir):
    # dir = "../data/tb"
    # dir = "../data/qa_data"
    # dir = "data"
    # dir = "../data/jd/big"

    counter = {}

    names = ["test", "valid", "train"]
    marks = ["src", "tgt", "attr"]
    for name in names:
        for mark in marks:
            source = mydir + "/" + name + "_" + mark + ".txt.untoken"
            target = mydir + "/" + name + "_" + mark + ".txt"
            # if mark == "attr":
            #     token_file(source=source, target=target, counter=None)
            # else:
            token_file(source=source, target=target, counter=counter)

    counter = dict(sorted(counter.items(), key=lambda kv: kv[1], reverse=True))

    with open(mydir + "/counter.json", "w", encoding="utf-8") as f:  # 特殊字符有问题，仅供人类阅读
        json.dump(counter, f, ensure_ascii=False)
    torch.save(counter, mydir + "/counter.bin")
    print("词频文件已经写入", mydir + "/counter.json", mydir + "/counter.bin")


def main(dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_dir', default=dir)
    parser.add_argument('-counter_path', default=dir + "/counter.bin")
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=30)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true', default=False)
    parser.add_argument('-share_vocab', action='store_true', default=True)
    parser.add_argument('-vocab', default=None)
    args = parser.parse_args()
    args.max_token_seq_len = args.max_word_seq_len + 2  # include the <s> and </s>

    word_count = torch.load(args.counter_path)
    word2idx, frequency = build_vocab_idx(word_count=word_count, min_word_count=args.min_word_count)
    vocab = {
        'settings': vars(args),
        'dict': {
            'src': word2idx,
            'tgt': word2idx,
            'ctx': word2idx,
            'frequency': frequency
        }}

    path = args.save_dir + "/reader.json"
    with open(path, "w", encoding="utf-8") as f:  # 特殊字符有问题，仅供人类阅读
        json.dump(vocab, f, ensure_ascii=False)
    print('[Info] 保存词汇到', os.path.abspath(path))

    path = args.save_dir + "/reader.data"
    print('[Info] 保存词汇到', os.path.abspath(path))
    torch.save(vocab, path)


if __name__ == '__main__':
    t0 = time()
    # mydir = "../data/jd/big"
    # mydir = "../data/jd/middle"
    mydir = "../data/jd/pure"
    count_file(mydir)
    main(mydir)
    print(time() - t0, "秒完成vocab.py")
