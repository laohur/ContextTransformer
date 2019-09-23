# import argparse
# import torch
# import os
# from time import time
# import random
from Util import *


def tokenize(line):
    # return list(line)
    # return line.split(" ")
    # words = line.split()
    # line = ''.join(line.split(" "))
    words = split_lans(line)
    # words = merge_gram(line)
    # line = ' '.join(words)
    # words = line.split(" ")
    re = []
    for word in words:
        if len(word) > 9:  # "13653923571"空值
            continue
        if word in [None, "", " ", "\t"]:
            continue
        re.append(word[:9])
    return re


def tb_pair(row):
    words = row.split("\t")
    # 数据多，出错丢弃.如果数据出错，前面几段都归为问题，最后一个归为回答.
    if len(words) < 3 or int(words[0]) != 1:
        return None, None
    question = words[1].strip()
    answer = words[2].strip()

    question = " ".join(tokenize(question))
    answer = " ".join(tokenize(answer))

    if len(question) < 2 or len(answer) < 2:
        print("tb_pair字太少", row)
        return None, None
    return question, answer


def get_jdpair(row):
    '''
        full_doc = ["skuid \t attributes \t review  \t counter \t question \t answer"]
    '''
    sents = row.split("\t")
    if len(sents) != 6 or int(sents[3]) < 20:  # 去除冷门商品
        return None, None, None
    # question = pure(sents[0].strip())
    # answer = pure(sents[1].strip())
    question = sents[4].strip()
    answer = sents[5].strip()
    attr = sents[1].strip()

    # question = tokenize(question)
    # answer = tokenize(answer)
    # attr = tokenize(attr)

    # if len(question) < 1 or len(answer) < 1:
    #     print("get_pair字太少", row)
    #     return None, None
    return question, answer, attr


def get_pair(row):
    sents = row.split("\t")
    if len(sents) < 2:
        return None, None
    # question = pure(sents[0].strip())
    # answer = pure(sents[1].strip())
    question = sents[0].strip()
    answer = sents[1].strip()

    question = " ".join(tokenize(question))
    answer = " ".join(tokenize(answer))

    # if len(question) < 1 or len(answer) < 1:
    #     print("get_pair字太少", row)
    #     return None, None
    return question, answer


def read(path, begin=0, end=-1):
    t0 = time()
    print("read正在读取", os.path.abspath(path))
    doc = open(path, "r", encoding="utf-8").read().splitlines()
    random.shuffle(doc)
    if end < 0:
        end = len(doc)
    print(time() - t0, "秒读出", len(doc), "条")
    t0 = time()
    qlenth, alenth, tlenth = 0, 0, 0  # 问题长度
    questions, answers, attrs = [], [], []
    for i in range(len(doc)):
        if i % 100000 == 0:
            print("进展", i * 100.0 / (end - begin), "读取第", i, "行，选取", len(questions))
        if i < begin:
            continue
        if i > end:
            break
        row = doc[i]
        # question, answer = tb_pair(row)
        # question, answer = get_pair(row)
        question, answer, attr = None, None, None
        try:
            question, answer, attr = get_jdpair(row)
        except Exception as e:
            print(e)
            continue
        valid = True
        for item in [question, answer, attr]:
            if item in [None, "", " "] or len(item) < 3 or len(item) > 80:  # big1 middle5
                valid = False
                break
        if not valid:
            continue
        if ('不' in answer) and random.random() < 0.4:
            continue
        if len(question) > 30 or len(answer) > 30 or len(attr) > 50:
            continue

        # if len(item) < 5 or len(item) > 60:
        #     continue

        qlenth += len(question)
        alenth += len(answer)
        tlenth += len(attr)
        questions.append(question)
        answers.append(answer)
        attrs.append(attr)
        # questions.append(' '.join(question))
        # answers.append(' '.join(answer))
        # attrs.append(' '.join(attr))

        if i % 100000 == 0:
            print(question, "--->", answer, "<---", attr)
            print("平均问题长", qlenth / len(questions), "平均回答长", alenth / len(answers), "平均详情长", tlenth / len(attrs))

    assert len(answers) == len(questions)
    print(str(path) + "总计", len(doc), "行，有效问答有" + str(len(answers)))
    print(time() - t0, "秒处理", len(answers), "条")
    return questions, answers, attrs


def splits_write(x, suffix, dir):  # 此处不能独自洗牌，应该对问答对洗牌
    print("splits_write正在划分训练集", os.path.abspath(dir))
    test_len, valid_len = 1000, 2000
    # right = len(x) - test_len
    # left = right - valid_len

    with open(dir + "/test" + suffix, "w", encoding="utf-8") as f:
        f.write("\n".join(x[:test_len]))
    print("测试集已写入", test_len)
    with open(dir + "/valid" + suffix, "w", encoding="utf-8") as f:
        f.write("\n".join(x[test_len:valid_len]))
    print("验证集已写入", valid_len - test_len)
    with open(dir + "/train" + suffix, "w", encoding="utf-8") as f:
        f.write("\n".join(x[valid_len:]))
    print("训练集已写入", len(x) - valid_len)

    print("训练集、验证集、测试集已写入", os.path.abspath(dir), "目录下")


def split_test(path, mydir):
    t0 = time()
    print("read正在读取", os.path.abspath(path))
    doc = open(path, "r", encoding="utf-8").read().splitlines()
    print(time() - t0, "秒读出", len(doc), "条")
    # for i in range(len(doc) - 1, 0, -1):
    #     if len(doc[i]) > 120:
    #         del doc[i]
    random.shuffle(doc)
    print(time() - t0, "秒 ", len(doc), "条")
    splits_write(doc, dir=mydir, suffix=".txt")


'''
说明太长，平均100， 如此挤占问答长度
4.639490365982056 秒读出 1685750 条
158.49600982666016 秒移除超过200字符后有 1062261 条
'''


def main(mydir):
    # dir = "../data/tb"
    # source = "/train.txt"

    # dir = "../data/qa_data"
    # source = "qa_data.txt"

    # dir = "../data/chitchat_data"
    # source = "chitchat_data.txt"

    # dir = "../data/all"
    # source = "balance.txt"

    dir = "../data/jd"
    source = "full.skuqa"

    split_test(dir + "/" + source, mydir)  #划分训练集

    names = ["test", "valid", "train"]
    marks = ["src", "tgt", "attr"]  # 拆分集合文件
    for name in names:
        result = read(mydir + "/" + name + ".txt", begin=0, end=-1)
        for i in range(3):
            path = mydir + "/" + name + "_" + marks[i] + ".txt.untoken"
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(result[i]))
            print(" 已写入", os.path.abspath(path))

if __name__ == '__main__':
    t0 = time()
    # mydir = "../data/jd/middle"
    mydir = "../data/jd/pure"
    main(mydir)
    print(time() - t0, "秒执行完main()")
