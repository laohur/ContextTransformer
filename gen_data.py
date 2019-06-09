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
    if len(sents) != 6 or int(sents[3]) < 20:  #去除冷门商品
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
            if item in [None, "", " "] or len(item) < 1:  # big
                valid = False
                break
        if not valid:
            continue

        # if len(question) > 25 or len(answer) > 25 or len(attr) > 50:
        #     continue

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

    mydir = "../data/jd/big"
    # mydir = "../data/jd/middle"
    split_test(dir + "/" + source, mydir)

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
C:\ProgramData\Anaconda3\python.exe "C:\Program Files\JetBrains\PyCharm Professional Edition with Anaconda plugin 2019.1.2\helpers\pydev\pydevconsole.py" --mode=client --port=53079
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\code\\ContextTransformer', 'D:/code/ContextTransformer'])
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.4.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 7.4.0
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/gen_data.py', wdir='D:/code/ContextTransformer')
read正在读取 D:\code\data\jd\full.skuqa
4.411203861236572 秒读出 1684340 条
5.808468341827393 秒  1684340 条
splits_write正在划分训练集 D:\code\data\jd\big
测试集已写入 1000
验证集已写入 1000
训练集已写入 1682340
训练集、验证集、测试集已写入 D:\code\data\jd\big 目录下
read正在读取 D:\code\data\jd\big\test.txt
0.002992391586303711 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
../data/jd/big/test.txt总计 1000 行，有效问答有363
0.0019941329956054688 秒处理 363 条
 已写入 D:\code\data\jd\big\test_src.txt.untoken
 已写入 D:\code\data\jd\big\test_tgt.txt.untoken
 已写入 D:\code\data\jd\big\test_attr.txt.untoken
read正在读取 D:\code\data\jd\big\valid.txt
0.0030231475830078125 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
这款可以同时连几部手机？连接不用配对密码的吗？ ---> 密码到不用，就是蓝牙信号不好。 <--- 三星（SAMSUNG）Level U 项圈式 运动蓝牙音乐耳机（雅墨黑）蓝牙耳机手机配件手机
平均问题长 23.0 平均回答长 15.0 平均详情长 46.0
../data/jd/big/valid.txt总计 1000 行，有效问答有360
0.001997232437133789 秒处理 360 条
 已写入 D:\code\data\jd\big\valid_src.txt.untoken
 已写入 D:\code\data\jd\big\valid_tgt.txt.untoken
 已写入 D:\code\data\jd\big\valid_attr.txt.untoken
read正在读取 D:\code\data\jd\big\train.txt
5.667839288711548 秒读出 1682340 条
进展 0.0 读取第 0 行，选取 0
进展 5.944101667914928 读取第 100000 行，选取 37416
进展 11.888203335829855 读取第 200000 行，选取 74610
用送的手机壳保护手机。会摔碎手机背面的玻璃吗。 ---> 不知道，没体验过哈哈 <--- 荣耀10 全面屏AI摄影手机 6GB+128GB 游戏手机 幻影紫 全网通 移动联通电信4G 双卡双待手机手机通讯手机
平均问题长 16.087239147042663 平均回答长 11.79468174933991 平均详情长 58.298521665706126
进展 17.832305003744786 读取第 300000 行，选取 111663
硬盘多大啊 ---> 256g <--- 宏碁(Acer)蜂鸟Swift3轻薄本 蓝朋友 14英寸全金属笔记本电脑SF314(i5-8250U 8G 256G SSD IPS win10)笔记本电脑整机电脑、办公
平均问题长 16.053687849262072 平均回答长 11.776794669723456 平均详情长 58.29751755265797
进展 23.77640667165971 读取第 400000 行，选取 148634
进展 29.72050833957464 读取第 500000 行，选取 185748
进展 35.66461000748957 读取第 600000 行，选取 222819
invalid literal for int() with base 10: ' counter '
进展 41.60871167540449 读取第 700000 行，选取 259867
进展 47.55281334331942 读取第 800000 行，选取 297131
怎么激活office？好难 ---> 买订阅账号 <--- 华硕顽石(ASUS) 五代FL8000UQ 15.6英寸影音笔记本电脑(i7-8550U 8G 128GSSD+1T 940MX 2G独显 FHD)星空灰笔记本电脑整机电脑、办公
平均问题长 16.043142441743065 平均回答长 11.73223011994669 平均详情长 58.285916696956235
进展 53.49691501123435 读取第 900000 行，选取 334195
进展 59.44101667914928 读取第 1000000 行，选取 371405
进展 65.3851183470642 读取第 1100000 行，选取 408256
进展 71.32922001497914 读取第 1200000 行，选取 445517
进展 77.27332168289406 读取第 1300000 行，选取 482600
进展 83.21742335080899 读取第 1400000 行，选取 519460
能自动清洁吗？ ---> 好像不能 <--- 格力（GREE）大1匹 变频 京慕 一级能效 京东微联 冷暖 空调挂机 KFR-26GW/NhEaB1W空调大 家 电家用电器
平均问题长 16.040499671775166 平均回答长 11.684438292768851 平均详情长 58.28394431920779
进展 89.16152501872392 读取第 1500000 行，选取 556807
可以直连ps4吗？ ---> 可以连，但是要转接头 <--- 戴尔（DELL） SE2416H 23.8英寸微边框 HDMI高清接口 广视角IPS屏 电脑显示器显示器电脑配件电脑、办公
平均问题长 16.040881237338542 平均回答长 11.68007284378098 平均详情长 58.28525631815635
进展 95.10562668663884 读取第 1600000 行，选取 593982
支持应用分身吗？ ---> 支持 <--- 荣耀 V10 高配版 6GB+64GB 幻夜黑 移动联通电信4G全面屏游戏手机 双卡双待手机手机通讯手机
平均问题长 16.042391785623494 平均回答长 11.677605251328742 平均详情长 58.28297442856109
../data/jd/big/train.txt总计 1682340 行，有效问答有624405
3.321113348007202 秒处理 624405 条
 已写入 D:\code\data\jd\big\train_src.txt.untoken
 已写入 D:\code\data\jd\big\train_tgt.txt.untoken
 已写入 D:\code\data\jd\big\train_attr.txt.untoken
25.03330111503601 秒执行完main()


    '''
