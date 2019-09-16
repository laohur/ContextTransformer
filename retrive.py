'''
检索答案
    full_doc = ["skuid \t attributes \t review  \t counter \t question \t answer"]

'''

import argparse
# import torch
# import transformer.Constants as Constants
# import json
# import math
# from time import time
# import os
# from Util import *
import os
import sys
from datetime import time
from bleu import bleu
from gen_data import *


def read_dict(path, begin=0, end=sys.maxsize):
    t0 = time()
    print("read正在读取", os.path.abspath(path))
    doc = open(path, "r", encoding="utf-8").read().splitlines()
    print(time() - t0, "秒读出", len(doc), "条")
    t0 = time()
    sku_dict = {}
    for i in range(len(doc)):
        if i < begin:
            continue
        if i > end:
            break
        sents = doc[i].split("\t")
        if sents[0] in sku_dict:
            sku_dict[sents[0]].append(sents[4:])
        else:
            sku_dict[sents[0]] = [sents[4:]]

    print(time() - t0, "秒拆开")
    return sku_dict


def read_qst(path):
    questions = []
    doc = open(path, "r", encoding="utf-8").read().splitlines()
    for line in doc:
        sents = line.split("\t")  # skuid attr review hot qs ans
        questions.append([sents[0], sents[4], sents[5]])
    return questions


def split_grams(line, n_gram=2):
    tokens = tokenize(line)
    grams = []
    for i in range(len(tokens) - n_gram):
        grams.append(''.join(tokens[i:i + n_gram]))
    if len(grams) == 0:
        grams.append(line)
    return grams


def split_grams0(line, n_gram=2):
    tokens = tokenize(line)
    grams = [tokens[0]]
    if len(tokens) <= 1:
        return grams
    for i in range(1, len(tokens)):
        grams.append(tokens[i - 1:i + 1])
    return grams


def cdrate(q, k, ngram=2):
    # comprehensive + determine -->  cdscore
    if (isinstance(q, str)):
        q = split_grams(q, ngram)
    if (isinstance(k, str)):
        k = split_grams(k, ngram)
    k = list(k)
    klen = len(k)
    if (klen == 0 or len(q) == 0):
        print("klen==0 or len(q)==0", k, q)
        return 0
    for gram in q:
        if gram in k:
            k.remove(gram)  # affect
        else:
            continue
    common = klen - len(k)
    # crate=common/len(q)
    # drate=common/klen
    # f1score=2*cdrate*drate/(cdrate+drate)
    score = 2.0 * common / (len(q) + klen)
    return score


def cdscore(hypothes, references, maxgrams=4):
    # 多个候选取平均
    bluescore = 0
    for hypoth in hypothes:
        score = 0
        for reference in references:
            avg_gram = 0
            for i in range(1, 1 + maxgrams):
                avg_gram += cdrate(hypoth, reference, i)
            avg_gram /= maxgrams
            if avg_gram > score:
                score = avg_gram
        bluescore += score
    bluescore /= len(hypothes)
    return bluescore


def nearest_asw(question, doc):  # question skuid q a
    re = ['no_question', 'no_answer', -1]  # lib_qs lib_ans score
    score = -1
    for pair in doc:  # q a
        tmp = cdrate(question, pair[0])
        if tmp > score:
            score = tmp
            re = pair + [score]
    return re


def answer(sku_dict, questions):
    n_answered, avg_score = 1, 0
    for i in range(len(questions)):  # modify var
        qpair = questions[i]  # skuid q a
        re = ['no_question', 'no_answer', -1]  # lib_qs lib_ans score
        if qpair[0] in sku_dict:
            re = nearest_asw(qpair[1], sku_dict[qpair[0]])
            if re[2] > 0:  # 大量不存在相似问题
                n_answered += 1
                avg_score += re[2]
        line = qpair[1:] + re[:-1] + [str(re[-1])]
        line = "\t".join(line)  # query_q query_a lib_q lib_a score
        questions[i] = line
    avg_score /= n_answered
    print("问题总计", len(questions), "可检索", n_answered, "平均分数", avg_score)
    return questions


def main():
    dir = "../data/jd/pure"
    sku_dict = read_dict(dir + "/train.txt")  # dict[id]=[q,a]
    questions = read_qst(dir + "/test.txt")  # [id,q]
    answers = answer(sku_dict, questions)
    with open("output/answers.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(answers))
    print()


def test():
    # hypotheses = ["The brown fox jumps over the dog 笑"]
    # references = ["The quick brown fox jumps over the lazy dog 笑"]
    hypotheses = ['吃 饱 撑 得 ， 估 计 你 女 朋 友 比 你 重 两 倍 。']
    references = ['吃 饱 撑 得 ， 估 计 你 女 朋 友 比 你 重 两 倍 。']
    # hypotheses = ["It is a guide to action which ensures that the military always obeys the commands of the party."]
    # references = ["It is a guide to action that ensures that the military will forever heed Party commands.","It is the guiding principle which guarantees the military forces always being under the command of the Party.","It is the practical guide for the army always to heed the directions of the party."]

    gram = 5
    for i in range(1, gram):
        cd_score = cdscore(hypotheses, references, i)
        print(" cdscore  ngram:" + str(i) + "-->" + str(cd_score))
    bleu_score = bleu(hypotheses, references)
    print(" bleu  score-->" + str(bleu_score))


if __name__ == '__main__':
    t0 = time()
    # main()
    test()
    print(time() - t0, "秒执行完main()")
