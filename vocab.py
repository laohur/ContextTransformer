"处理文本"
import argparse
import torch
import transformer.Constants as Constants
import json
import math
from time import time
import os
from Util import merge_gram
import sys


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

    frequency = {}
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
            if mark == "attr":
                token_file(source=source, target=target, counter=None)
            else:
                token_file(source=source, target=target, counter=counter)

    counter = dict(sorted(counter.items(), key=lambda kv: kv[1], reverse=True))

    with open(mydir + "/counter.json", "w", encoding="utf-8") as f:  # 特殊字符有问题，仅供人类阅读
        json.dump(counter, f, ensure_ascii=False)
    torch.save(counter, mydir + "/counter.bin")
    print("词频文件已经写入", mydir + "/counter.json", mydir + "/counter.bin")


def main(dir):
    # dir = "../data/jd/middle"

    parser = argparse.ArgumentParser()
    parser.add_argument('-save_dir', default=dir)
    parser.add_argument('-counter_path', default=dir + "/counter.bin")
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=20)
    parser.add_argument('-min_word_count', type=int, default=9)
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
    mydir = "../data/jd/middle"
    # mydir = "../data/jd/big"
    # count_file(mydir)
    main(mydir)
    print(time() - t0, "秒完成vocab.py")
'''C:\ProgramData\Anaconda3\python.exe "C:\Program Files\JetBrains\PyCharm Professional Edition with Anaconda plugin 2019.1.2\helpers\pydev\pydevconsole.py" --mode=client --port=51938
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\code\\ContextTransformer', 'D:/code/ContextTransformer'])
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.4.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 7.4.0
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/vocab.py', wdir='D:/code/ContextTransformer')
token_file分词 D:\code\data\jd\middle\test_src.txt.untoken 将写入 D:\code\data\jd\middle\test_src.txt
第一条 衣服是纯棉的么
../data/jd/middle/test_src.txt.untoken 已全部分词至 ../data/jd/middle/test_src.txt
token_file分词 D:\code\data\jd\middle\test_tgt.txt.untoken 将写入 D:\code\data\jd\middle\test_tgt.txt
第一条 是纯棉的
../data/jd/middle/test_tgt.txt.untoken 已全部分词至 ../data/jd/middle/test_tgt.txt
token_file分词 D:\code\data\jd\middle\test_attr.txt.untoken 将写入 D:\code\data\jd\middle\test_attr.txt
../data/jd/middle/test_attr.txt.untoken 不计入词频
第一条 童亦 0-8个月 初生 婴儿衣服婴儿礼盒 定性枕 抱被 春夏新生儿礼盒套装宝宝用品刚出生满月礼物婴儿礼盒童装母婴
../data/jd/middle/test_attr.txt.untoken 已全部分词至 ../data/jd/middle/test_attr.txt
token_file分词 D:\code\data\jd\middle\valid_src.txt.untoken 将写入 D:\code\data\jd\middle\valid_src.txt
第一条 请问NFC怎么用？周围没人用，帮忙科普一下，谢谢啦。
../data/jd/middle/valid_src.txt.untoken 已全部分词至 ../data/jd/middle/valid_src.txt
token_file分词 D:\code\data\jd\middle\valid_tgt.txt.untoken 将写入 D:\code\data\jd\middle\valid_tgt.txt
第一条 可以当公交卡银行卡使用
../data/jd/middle/valid_tgt.txt.untoken 已全部分词至 ../data/jd/middle/valid_tgt.txt
token_file分词 D:\code\data\jd\middle\valid_attr.txt.untoken 将写入 D:\code\data\jd\middle\valid_attr.txt
../data/jd/middle/valid_attr.txt.untoken 不计入词频
第一条 小米Note3 美颜双摄拍照手机 6GB+64GB 黑色 全网通4G手机 双卡双待手机手机通讯手机
../data/jd/middle/valid_attr.txt.untoken 已全部分词至 ../data/jd/middle/valid_attr.txt
token_file分词 D:\code\data\jd\middle\train_src.txt.untoken 将写入 D:\code\data\jd\middle\train_src.txt
第一条 呵护是
第 100000 行 你这种可以把皮肤变白吗？
 ---> 你 这 种 可 以 把 皮 肤 变 白 吗 ？ 
已统计词频 6508
第 200000 行 我的华为手机怎么打不开视频
 ---> 我 的 华 为 手 机 怎 么 打 不 开 视 频 
已统计词频 8110
第 300000 行 同事推荐我来买修正的，说是老牌子比较好？祛痘印的效果怎么样？
 ---> 同 事 推 荐 我 来 买 修 正 的 ， 说 是 老 牌 子 比 较 好 ？ 祛 痘 印 的 效 果 怎 么 样 ？ 
已统计词频 9296
../data/jd/middle/train_src.txt.untoken 已全部分词至 ../data/jd/middle/train_src.txt
token_file分词 D:\code\data\jd\middle\train_tgt.txt.untoken 将写入 D:\code\data\jd\middle\train_tgt.txt
第一条 高端的级别吧
第 100000 行 长期用长期用有效果的！
 ---> 长 期 用 长 期 用 有 效 果 的 ！ 
已统计词频 10680
第 200000 行 小米手机自带app
 ---> 小 米 手 机 自 带 app 
已统计词频 11619
第 300000 行 有效果
 ---> 有 效 果 
已统计词频 12370
../data/jd/middle/train_tgt.txt.untoken 已全部分词至 ../data/jd/middle/train_tgt.txt
token_file分词 D:\code\data\jd\middle\train_attr.txt.untoken 将写入 D:\code\data\jd\middle\train_attr.txt
../data/jd/middle/train_attr.txt.untoken 不计入词频
第一条 伊利奶粉 金领冠珍护系列 幼儿配方奶粉 3段900克（1-3岁幼儿适用）新老包装随机发货婴幼奶粉奶粉母婴
第 100000 行 【直降50】膜法世家面膜美白去黑头清洁控油补水淡化痘印收缩毛孔水洗绿豆泥浆面膜男女士护肤品面膜面部护肤美妆护肤
 ---> 【 直 降 50 】 膜 法 世 家 面 膜 美 白 去 黑 头 清 洁 控 油 补 水 淡 化 痘 印 收 缩 毛 孔 水 洗 绿 豆 泥 浆 面 膜 男 女 士 护 肤 品 面 膜 面 部 护 肤 美 妆 护 肤 
第 200000 行 小白智能摄像头云台1080p版无线wifi监控高清智能摄像机室内外家用办公360°红外夜视摄像头支持小米路由器智能家居智能设备数码
 ---> 小 白 智 能 摄 像 头 云 台 1080 p 版 无 线 wifi 监 控 高 清 智 能 摄 像 机 室 内 外 家 用 办 公 360 ° 红 外 夜 视 摄 像 头 支 持 小 米 路 由 器 智 能 家 居 智 能 设 备 数 码 
第 300000 行 修正积雪草祛痘印淡化膏去痘痘坑痘疤修复凹洞霜产品前女五男士强 一盒 20g乳液/面霜面部护肤美妆护肤
 ---> 修 正 积 雪 草 祛 痘 印 淡 化 膏 去 痘 痘 坑 痘 疤 修 复 凹 洞 霜 产 品 前 女 五 男 士 强 一 盒 20 g 乳 液 / 面 霜 面 部 护 肤 美 妆 护 肤 
../data/jd/middle/train_attr.txt.untoken 已全部分词至 ../data/jd/middle/train_attr.txt
词频文件已经写入 ../data/jd/middle/counter.json ../data/jd/middle/counter.bin
55.69700527191162 秒完成vocab.py
'''
'''[Info] 原始词库 = 12613
[Info] 频繁字典大小 = 4475, 最低频数 = 9
[Info] 忽略罕词数 = 7233 爆表词汇数 -87387
[Info] 保存词汇到 D:\code\data\jd\middle\reader.json
[Info] 保存词汇到 D:\code\data\jd\middle\reader.data
0.12566924095153809 秒完成vocab.py'''