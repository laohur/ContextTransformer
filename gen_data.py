import argparse
import torch
import os
from time import time
import random
from Util import merge_gram


def tokenize(line):
    # return list(line)
    # return line.split(" ")
    # words = line.split()
    # line = ''.join(line.split(" "))
    # words = split_lans(line)
    # words = merge_gram(line)
    # line = ' '.join(words)
    words = line.split(" ")
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
    if len(sents) != 6:
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
            if item in [None, "", " "] or len(item) < 5:
                valid = False
                break
        if not valid:
            continue
        if len(question) > 25 or len(answer) > 25 or len(attr) > 50:
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

    # mydir = "../data/jd/big"
    mydir = "../data/jd/middle"
    # split_test(dir + "/" + source, mydir)

    names = ["test", "valid", "train"]
    marks = ["src", "tgt", "attr"]
    for name in names:
        result = read(mydir + "/" + name + ".txt", begin=0, end=-1)
        for i in range(3):
            path = mydir + "/" + name + "_" + marks[i] + ".txt.untoken"
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
C:\ProgramData\Anaconda3\python.exe "C:\Program Files\JetBrains\PyCharm Professional Edition with Anaconda plugin 2019.1.2\helpers\pydev\pydevconsole.py" --mode=client --port=50175
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\code\\ContextTransformer', 'D:/code/ContextTransformer'])
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.4.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 7.4.0
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/gen_data.py', wdir='D:/code/ContextTransformer')
read正在读取 D:\code\data\jd\full.skuqa
4.704388856887817 秒读出 1684340 条
6.784824371337891 秒  1684340 条
splits_write正在划分训练集 D:\code\data\jd\middle
测试集已写入 1000
验证集已写入 1000
训练集已写入 1682340
训练集、验证集、测试集已写入 D:\code\data\jd\middle 目录下


C:\ProgramData\Anaconda3\python.exe "C:\Program Files\JetBrains\PyCharm Professional Edition with Anaconda plugin 2019.1.2\helpers\pydev\pydevconsole.py" --mode=client --port=50833
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\code\\ContextTransformer', 'D:/code/ContextTransformer'])
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.4.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 7.4.0
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/gen_data.py', wdir='D:/code/ContextTransformer')
read正在读取 D:\code\data\jd\middle\test.txt
0.003988981246948242 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
../data/jd/middle/test.txt总计 1000 行，有效问答有188
0.002111196517944336 秒处理 188 条
 已写入 D:\code\data\jd\middle\test_src.txt.untoken
 已写入 D:\code\data\jd\middle\test_tgt.txt.untoken
 已写入 D:\code\data\jd\middle\test_attr.txt.untoken
read正在读取 D:\code\data\jd\middle\valid.txt
0.0029582977294921875 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
麻烦你说一下你们的联系方式谢谢 ---> 什么意思？ <--- 杨红樱淘气包马小跳系列（典藏版 套装10册）儿童文学童书图书
平均问题长 15.0 平均回答长 5.0 平均详情长 30.0
../data/jd/middle/valid.txt总计 1000 行，有效问答有194
0.002992391586303711 秒处理 194 条
 已写入 D:\code\data\jd\middle\valid_src.txt.untoken
 已写入 D:\code\data\jd\middle\valid_tgt.txt.untoken
 已写入 D:\code\data\jd\middle\valid_attr.txt.untoken
read正在读取 D:\code\data\jd\middle\train.txt
6.360016584396362 秒读出 1682340 条
进展 0.0 读取第 0 行，选取 0
进展 5.944101667914928 读取第 100000 行，选取 19332
进展 11.888203335829855 读取第 200000 行，选取 38799
进展 17.832305003744786 读取第 300000 行，选取 58019
进展 23.77640667165971 读取第 400000 行，选取 77513
你的颜色对版吗？ ---> 没开花，不知道呢， <--- 春天来了   盆栽蟹爪兰花苗 多肉植物 蟹爪兰带根发货当年开花办公桌阳台植物 苗木花卉绿植农资绿植
平均问题长 11.461542431044714 平均回答长 11.310331036973967 平均详情长 42.42900637304229
进展 29.72050833957464 读取第 500000 行，选取 97146
进展 35.66461000748957 读取第 600000 行，选取 116517
这个一天吃几顿啊？ ---> 我当晚餐，两包，加点芝麻糊。 <--- 明安旭 魔芋代餐粉 代餐饱腹食品早晚代餐魔芋粉 500g大容量内含40小包冲饮谷物饮料冲调食品饮料
平均问题长 11.457706105494431 平均回答长 11.32813814174634 平均详情长 42.44515868792804
进展 41.60871167540449 读取第 700000 行，选取 135746
进展 47.55281334331942 读取第 800000 行，选取 155207
生产日期？ ---> 看产品批次吧，问问客服 <--- 光明 莫斯利安 常温酸牛奶（原味）200g*24家庭装中华老字号牛奶乳品饮料冲调食品饮料
平均问题长 11.454757486727488 平均回答长 11.32404901809185 平均详情长 42.43703288490284
进展 53.49691501123435 读取第 900000 行，选取 174626
进展 59.44101667914928 读取第 1000000 行，选取 194048
进展 65.3851183470642 读取第 1100000 行，选取 213273
味道浓吗，喷多少次才能闻香识人 ---> 留香时间很长吧 <--- 爱马仕（HERMES） 香水女士男士淡香水持久香氛香水香水彩妆美妆护肤
平均问题长 11.459779438656376 平均回答长 11.320671999399833 平均详情长 42.437746748314375
进展 71.32922001497914 读取第 1200000 行，选取 232789
进展 77.27332168289406 读取第 1300000 行，选取 252229
近视眼取下眼镜可以用吗 ---> 不可以，没有近视度数调解功能，可以直接戴眼镜使用 <--- 小鸟看看 Pico Neo VR一体机 基础版VR眼镜智能设备数码
平均问题长 11.463124925663084 平均回答长 11.321551758315822 平均详情长 42.43556674463783
进展 83.21742335080899 读取第 1400000 行，选取 271814
进展 89.16152501872392 读取第 1500000 行，选取 291311
进展 95.10562668663884 读取第 1600000 行，选取 310690
../data/jd/middle/train.txt总计 1682340 行，有效问答有326702
3.994313955307007 秒处理 326702 条
 已写入 D:\code\data\jd\middle\train_src.txt.untoken
 已写入 D:\code\data\jd\middle\train_tgt.txt.untoken
 已写入 D:\code\data\jd\middle\train_attr.txt.untoken
11.53410792350769 秒执行完main()



    '''
