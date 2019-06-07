import argparse
import torch
import os
from time import time
import random
from Util import *


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


def tokenize(line):
    # return list(line)
    # return line.split(" ")
    # words = line.split()
    # line = ''.join(line.split(" "))
    # words = split_lans(line)
    words = merge_gram(line)
    # line = ' '.join(words)
    # words = line.split(" ")
    re = []
    for word in words:
        if len(word) > 7:
            continue
        if word:
            re.append(word)
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
    if len(sents) != 6:
        return None, None, None
    # question = pure(sents[0].strip())
    # answer = pure(sents[1].strip())
    question = sents[4].strip()
    answer = sents[5].strip()
    attr = sents[1].strip()

    question = " ".join(tokenize(question))
    answer = " ".join(tokenize(answer))
    attr = " ".join(tokenize(attr))

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
        question, answer, attr = get_jdpair(row)
        valid = True
        for item in [question, answer, attr]:
            if item in [None, "", " "] or len(item) < 5 or len(item) > 80:
                valid = False
                break
        if not valid:
            continue

        # if question == None or answer == None or attr == None:
        #     continue
        # if len(question)<10 or len(attr)<10 or len(answer)<10

        qlenth += len(question)
        alenth += len(answer)
        tlenth += len(attr)
        questions.append(question)
        answers.append(answer)
        attrs.append(attr)

        if i % 100000 == 0:
            print(question, "--->", answer, "<---", attr)
            print("平均问题长", qlenth / len(questions), "平均回答长", alenth / len(answers), "平均详情长", tlenth / len(attrs))

    assert len(answers) == len(questions)
    print(str(path) + "总计", len(doc), "行，有效问答有" + str(len(answers)))
    print(time() - t0, "秒处理", len(answers), "条")
    return questions, answers, attrs


def split_test(path):
    t0 = time()
    print("read正在读取", os.path.abspath(path))
    doc = open(path, "r", encoding="utf-8").read().splitlines()
    print(time() - t0, "秒读出", len(doc), "条")
    # for i in range(len(doc) - 1, 0, -1):
    #     if len(doc[i]) > 120:
    #         del doc[i]
    random.shuffle(doc)
    print(time() - t0, "秒 ", len(doc), "条")
    splits_write(doc, dir="data", suffix=".txt")


'''
说明太长，平均100， 如此挤占问答长度
4.639490365982056 秒读出 1685750 条
158.49600982666016 秒移除超过200字符后有 1062261 条
'''


def main():
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

    split_test(dir + "/" + source)
    names = ["train", "valid", "test"]
    marks = ["src", "tgt", "attr"]
    for name in names:
        result = read("data/" + name + ".txt", begin=0, end=-1)
        for i in range(3):
            path = "data/" + name + "_" + marks[i] + ".txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(result[i]))
            print(" 已写入", os.path.abspath(path))

    # questions, answers, attrs = read(dir + "/" + source, begin=0, end=-1)
    # splits_write(questions, dir="data", suffix="_src.txt")
    # splits_write(answers, dir="data", suffix="_tgt.txt")
    # splits_write(attrs, dir="data", suffix="_attr.txt")


if __name__ == '__main__':
    t0 = time()
    main()
    print(time() - t0, "秒执行完main()")

'''
../data/jd//full.skuqa总计 1685750 行，有效问答有300001
平均问题长 27.155192816023945 平均回答长 21.655607814640618 平均详情长 176.3915820280599
27.85947823524475 秒处理 300001 条

C:\ProgramData\Anaconda3\python.exe D:/code/qa_chat_jd3/gen_data.py
read正在读取 D:\code\data\jd\full.skuqa
4.701453685760498 秒读出 1684340 条
6.367985725402832 秒  1684340 条
splits_write正在划分训练集 D:\code\qa_chat_jd3\data
测试集已写入
验证集已写入
训练集、验证集、测试集已写入 D:\code\qa_chat_jd3\data 目录下
read正在读取 D:\code\qa_chat_jd3\data\train.txt
5.6169798374176025 秒读出 1682340 条
进展 0.0 读取第 0 行，选取 0
进展 5.944101667914928 读取第 100000 行，选取 20802
进展 11.888203335829855 读取第 200000 行，选取 41721
进展 17.832305003744786 读取第 300000 行，选取 62654
进展 23.77640667165971 读取第 400000 行，选取 83456
进展 29.72050833957464 读取第 500000 行，选取 104045
稳 当 吗 ， 会 摇 晃 吗 ？ ---> 稳 当 ！ <--- 婴 儿 推 车 轻 便 折 叠 婴 儿 车 伞 车 可 坐 可 平 躺 便 携 避 震 宝 宝 手 推 车 婴 儿 推 车 童 车 童 床 母 婴
平均问题长 24.286565557541856 平均回答长 20.628279799319532 平均详情长 68.52857390000577
进展 35.66461000748957 读取第 600000 行，选取 124865
进展 41.60871167540449 读取第 700000 行，选取 145997
能 开 椰 子 盖 吗 ---> 没 问 题 的 <--- 华 丰 巨 箭 HF - 6311140 手 板 锯 400 MM 65 锰 钢 手 锯 木 工 手 钢 锯 子 手 动 工 具 五 金 工 具 家 装 建 材
平均问题长 24.266250222605787 平均回答长 20.605501445225276 平均详情长 68.53596624611296
进展 47.55281334331942 读取第 800000 行，选取 167390
如 果 货 到 付 款 我 不 在 怎 么 办 ---> 联 系 不 到 你 ， 商 品 会 退 回 <--- 荣 耀 畅 玩 7 X 4 GB + 32 GB 全 网 通 4 G 全 面 屏 手 机 标 配 版 魅 焰 红 手 机 手 机 通 讯 手 机
平均问题长 24.264357104025905 平均回答长 20.612535918896477 平均详情长 68.51453781864019
进展 53.49691501123435 读取第 900000 行，选取 188394
进展 59.44101667914928 读取第 1000000 行，选取 209510
进展 65.3851183470642 读取第 1100000 行，选取 230703
进展 71.32922001497914 读取第 1200000 行，选取 251484
新 轩 逸 可 以 用 不 ？ ---> 不 知 道 ， 反 正 我 的 英 菲 尼 迪 QX 70 用 着 挺 好 的 。 <--- 东 风 嘉 实 多 佳 驰 黑 佳 驰 全 合 成 机 油 润 滑 油 5 W - 30 SN 级 4 L 汽 机 油 维 修 保 养 汽 车 用 品
平均问题长 24.295958009424023 平均回答长 20.637282541702287 平均详情长 68.51731514802076
进展 77.27332168289406 读取第 1300000 行，选取 272495
进展 83.21742335080899 读取第 1400000 行，选取 293336
进展 89.16152501872392 读取第 1500000 行，选取 314307
进展 95.10562668663884 读取第 1600000 行，选取 335118
郑 州 高 新 区 的 有 拼 单 的 吗 ？ ---> 没 有 了 <--- 五 羊 （ ） 瞬 吸 棉 柔 婴 儿 成 长 裤 加 加 大 号 XXL 76 片 【 15 kg 以 上 】 拉 拉 裤 尿 裤 湿 巾 母 婴
平均问题长 24.290896666557252 平均回答长 20.629104885130356 平均详情长 68.53222288202102
data/train.txt总计 1682340 行，有效问答有352246
170.90122413635254 秒处理 352246 条
 已写入 D:\code\qa_chat_jd3\data\train_src.txt
 已写入 D:\code\qa_chat_jd3\data\train_tgt.txt
 已写入 D:\code\qa_chat_jd3\data\train_attr.txt
read正在读取 D:\code\qa_chat_jd3\data\valid.txt
0.00399017333984375 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
data/valid.txt总计 1000 行，有效问答有188
0.10372161865234375 秒处理 188 条
 已写入 D:\code\qa_chat_jd3\data\valid_src.txt
 已写入 D:\code\qa_chat_jd3\data\valid_tgt.txt
 已写入 D:\code\qa_chat_jd3\data\valid_attr.txt
read正在读取 D:\code\qa_chat_jd3\data\test.txt
0.003980875015258789 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
data/test.txt总计 1000 行，有效问答有216
0.1047201156616211 秒处理 216 条
 已写入 D:\code\qa_chat_jd3\data\test_src.txt
 已写入 D:\code\qa_chat_jd3\data\test_tgt.txt
 已写入 D:\code\qa_chat_jd3\data\test_attr.txt
191.67412662506104 秒执行完main()

Process finished with exit code 0

'''
