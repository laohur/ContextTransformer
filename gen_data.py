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

'''
C:\ProgramData\Anaconda3\python.exe "C:\Program Files\JetBrains\PyCharm Professional Edition with Anaconda plugin 2019.1.2\helpers\pydev\pydevconsole.py" --mode=client --port=54279
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\code\\ContextTransformer', 'D:/code/ContextTransformer'])
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.4.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 7.4.0
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/gen_data.py', wdir='D:/code/ContextTransformer')
read正在读取 D:\code\data\jd\pure\test.txt
0.11268019676208496 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
../data/jd/pure/test.txt总计 1000 行，有效问答有169
0.003991842269897461 秒处理 169 条
 已写入 D:\code\data\jd\pure\test_src.txt.untoken
 已写入 D:\code\data\jd\pure\test_tgt.txt.untoken
 已写入 D:\code\data\jd\pure\test_attr.txt.untoken
read正在读取 D:\code\data\jd\pure\valid.txt
0.12866497039794922 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
../data/jd/pure/valid.txt总计 1000 行，有效问答有172
0.0029833316802978516 秒处理 172 条
 已写入 D:\code\data\jd\pure\valid_src.txt.untoken
 已写入 D:\code\data\jd\pure\valid_tgt.txt.untoken
 已写入 D:\code\data\jd\pure\valid_attr.txt.untoken
read正在读取 D:\code\data\jd\pure\train.txt
11.33012580871582 秒读出 1682340 条
进展 0.0 读取第 0 行，选取 0
进展 5.944101667914928 读取第 100000 行，选取 16931
进展 11.888203335829855 读取第 200000 行，选取 33768
进展 17.832305003744786 读取第 300000 行，选取 50467
请问西芹可以榨吗？ ---> 西芹可以榨，但是出汁会很少的 <--- 苏泊尔（SUPOR）榨汁机 大口径家用果汁机 TJE06D榨汁机/原汁机厨房小电家用电器
平均问题长 15.970159308868986 平均回答长 12.4876753586431 平均详情长 55.828386304192755
进展 23.77640667165971 读取第 400000 行，选取 67142
进展 29.72050833957464 读取第 500000 行，选取 83881
进展 35.66461000748957 读取第 600000 行，选取 100744
进展 41.60871167540449 读取第 700000 行，选取 117694
进展 47.55281334331942 读取第 800000 行，选取 134579
进展 53.49691501123435 读取第 900000 行，选取 151363
想问一问，此款手机在国外售价多少？ ---> 10000以上 <--- 华为 HUAWEI Mate RS 保时捷设计全网通版6G+256G 玄黑色 移动联通电信4G手机 双卡双待手机手机通讯手机
平均问题长 15.97451177294469 平均回答长 12.482380222509976 平均详情长 55.80570016648608
进展 59.44101667914928 读取第 1000000 行，选取 168223
invalid literal for int() with base 10: ' counter '
进展 65.3851183470642 读取第 1100000 行，选取 185236
进展 71.32922001497914 读取第 1200000 行，选取 201859
进展 77.27332168289406 读取第 1300000 行，选取 218737
进展 83.21742335080899 读取第 1400000 行，选取 235551
进展 89.16152501872392 读取第 1500000 行，选取 252449
进展 95.10562668663884 读取第 1600000 行，选取 269571
请问不连接歌华有线，仅连接WIFI可以看到电视直播吗？ ---> 应该可以，功能很多 <--- 小米（MI）小米电视4A 43英寸 L43M5-AZ 2GB+8GB HDR 全高清 人工智能网络液晶平板电视平板电视大 家 电家用电器
平均问题长 15.997959728755212 平均回答长 12.49629783508673 平均详情长 55.82591292864244
../data/jd/pure/train.txt总计 1682340 行，有效问答有283696
4.279518365859985 秒处理 283696 条
 已写入 D:\code\data\jd\pure\train_src.txt.untoken
 已写入 D:\code\data\jd\pure\train_tgt.txt.untoken
 已写入 D:\code\data\jd\pure\train_attr.txt.untoken
17.10163140296936 秒执行完main()
10万
'''