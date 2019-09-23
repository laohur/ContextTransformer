import argparse
import torch
import os
from time import time
import math
import random
import os


def tokenize(line):
    return line.split(" ")


def read(path, begin=0, end=math.inf, ):
    t0 = time()
    print("read正在读取", os.path.abspath(path), begin, "->", end)
    doc = open(path, "r", encoding="utf-8").read().splitlines()
    print(time() - t0, "秒读出", len(doc), "条")
    # random.shuffle(doc)
    # line_count = 0  #总行数
    qlenth, alenth = 0, 0  # 问题长度
    questions, answers = [], []

    for i in range(len(doc)):
        if i < begin:
            continue
        if i > end:
            break

        row = doc[i]

        sents = row.split("\t")
        # 数据多，出错丢弃.如果数据出错，前面几段都归为问题，最后一个归为回答.
        if len(sents) != 2:
            print("非一问一答，丢弃", sents)
            continue
        question = " ".join(tokenize(sents[0]))
        answer = " ".join(tokenize(sents[1]))

        qlenth += len(question)
        alenth += len(answer)
        questions.append(question)
        answers.append(answer)
        if i % 1000000 == 0:
            print(i / len(doc), "正在处理", i, "行", question, answer)

    assert len(answers) == len(questions)
    print("行，有效问答有" + str(len(answers)))
    print("平均问题长", qlenth / len(answers), "平均回答长", alenth / len(answers))
    return questions, answers


def splits_write(x, suffix, dir, shuffle=True):
    print("splits_write正在划分训练集", os.path.abspath(dir))
    if shuffle:
        random.shuffle(x)
    test_len, valid_len = 100, 1000
    right = len(x) - test_len
    left = right - valid_len

    with open(dir + "/test" + suffix, "w", encoding="utf-8") as f:
        f.write("\n".join(x[:test_len]))
    print("测试集已写入")
    with open(dir + "/valid" + suffix, "w", encoding="utf-8") as f:
        f.write("\n".join(x[test_len:valid_len]))
    print("验证集已写入")
    with open(dir + "/train" + suffix, "w", encoding="utf-8") as f:
        f.write("\n".join(x[valid_len:]))
    print("训练集、验证集、测试集已写入", dir, "目录下")


def main():
    dir = "../../data/qa_data"
    source = "qa_data.txt"
    questions, answers = read(dir + "/" + source)
    splits_write(questions, dir=dir, suffix="_src.txt")
    splits_write(answers, dir=dir, suffix="_tgt.txt")


if __name__ == '__main__':
    main()
