'''
检索答案
    full_doc = ["skuid \t attributes \t review  \t counter \t question \t answer"]

'''

import argparse
import torch
import transformer.Constants as Constants
import json
import math
from time import time
import os
from Util import *
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
        sents = line.split("\t")
        questions.append([sents[0], sents[4]])
    return questions


def split_grams(line, n_gram=2):
    tokens = tokenize(line)
    grams = [tokens[0]]
    if len(tokens) <= 1:
        return grams
    for i in range(1, len(tokens)):
        grams.append(tokens[i - 1:i + 1])
    return grams


def relation(q, k):
    q = split_grams(q)
    k = split_grams(k)
    count = len(k)
    for gram in q:
        if gram in k:
            k.remove(gram)
        else:
            continue
    score = 2.0 * (count - len(k)) / (len(q) + count)
    return score


def nearest_asw(question, doc):
    answer, score = 'not_found_skuid', 0
    for pair in doc:
        tmp = relation(question, pair[0])
        if tmp > score:
            score = tmp
            answer = pair[1]
    return answer, score


def answer(sku_dict, questions):
    n_answered, avg_score = 1, 0
    for i in range(len(questions)):
        qs = questions[i]
        if qs[0] in sku_dict:
            answer, score = nearest_asw(qs[1], sku_dict[qs[0]])
            n_answered += 1
            avg_score += score
        else:
            answer, score = 'not_found_skuid', 0
        qs = [answer, str(score)]
        questions[i] = "\t".join(qs)
    avg_score /= n_answered
    print("问题总计", len(questions), "可检索", n_answered, "平均分数", avg_score)
    return questions


def main():
    # dir = "../data/tb"
    # dir = "../data/qa_data"
    dir = "data"
    sku_dict = read_dict(dir + "/train.txt")  # dict[id]=[q,a]
    questions = read_qst(dir + "/test.txt")  # [id,q]
    answers = answer(sku_dict, questions)
    with open("data/answers.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(answers))
    print()


if __name__ == '__main__':
    t0 = time()
    main()
    print(time() - t0, "秒执行完main()")
