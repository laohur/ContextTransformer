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


def count_file():
    # dir = "../data/tb"
    # dir = "../data/qa_data"
    # dir = "data"
    # dir = "../data/jd/big"

    mydir = "../data/jd/middle"
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


def main():
    dir = "../data/jd/middle"

    parser = argparse.ArgumentParser()
    parser.add_argument('-save_dir', default=dir)
    parser.add_argument('-counter_path', default=dir + "/counter.bin")
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=20)
    parser.add_argument('-min_word_count', type=int, default=2)
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
    # count_file()
    main()
    print(time() - t0, "秒完成vocab.py")

'''
C:\ProgramData\Anaconda3\python.exe "C:\Program Files\JetBrains\PyCharm Professional Edition with Anaconda plugin 2019.1.2\helpers\pydev\pydevconsole.py" --mode=client --port=50942
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\code\\ContextTransformer', 'D:/code/ContextTransformer'])
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.4.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 7.4.0
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/vocab.py', wdir='D:/code/ContextTransformer')
token_file分词 D:\code\data\jd\middle\test_src.txt.untoken 将写入 D:\code\data\jd\middle\test_src.txt
第一条 带垫子吗？
../data/jd/middle/test_src.txt.untoken 已全部分词至 ../data/jd/middle/test_src.txt
token_file分词 D:\code\data\jd\middle\test_tgt.txt.untoken 将写入 D:\code\data\jd\middle\test_tgt.txt
第一条 不带垫子。
../data/jd/middle/test_tgt.txt.untoken 已全部分词至 ../data/jd/middle/test_tgt.txt
token_file分词 D:\code\data\jd\middle\test_attr.txt.untoken 将写入 D:\code\data\jd\middle\test_attr.txt
../data/jd/middle/test_attr.txt.untoken 不计入词频
第一条 叠亿 铁艺床 铁架床 铁床 双人床 卧室床 卧室家具 龙骨床 单床 不含床垫 238 床卧室家具家具
../data/jd/middle/test_attr.txt.untoken 已全部分词至 ../data/jd/middle/test_attr.txt
token_file分词 D:\code\data\jd\middle\valid_src.txt.untoken 将写入 D:\code\data\jd\middle\valid_src.txt
第一条 麻烦你说一下你们的联系方式谢谢
../data/jd/middle/valid_src.txt.untoken 已全部分词至 ../data/jd/middle/valid_src.txt
token_file分词 D:\code\data\jd\middle\valid_tgt.txt.untoken 将写入 D:\code\data\jd\middle\valid_tgt.txt
第一条 什么意思？
../data/jd/middle/valid_tgt.txt.untoken 已全部分词至 ../data/jd/middle/valid_tgt.txt
token_file分词 D:\code\data\jd\middle\valid_attr.txt.untoken 将写入 D:\code\data\jd\middle\valid_attr.txt
../data/jd/middle/valid_attr.txt.untoken 不计入词频
第一条 杨红樱淘气包马小跳系列（典藏版 套装10册）儿童文学童书图书
../data/jd/middle/valid_attr.txt.untoken 已全部分词至 ../data/jd/middle/valid_attr.txt
token_file分词 D:\code\data\jd\middle\train_src.txt.untoken 将写入 D:\code\data\jd\middle\train_src.txt
第一条 这个灯光是什么颜色的，冲一次电能用多久
第 100000 行 身高170.体重77穿多大的
 ---> 身 高 170 . 体 重 77 穿 多 大 的 
已统计词频 5541
第 200000 行 声音大吗？
 ---> 声 音 大 吗 ？ 
已统计词频 6847
第 300000 行 这烤肠好吃吗，辣吗，和良品铺子的烤肠哪个好吃
 ---> 这 烤 肠 好 吃 吗 ， 辣 吗 ， 和 良 品 铺 子 的 烤 肠 哪 个 好 吃 
已统计词频 7748
../data/jd/middle/train_src.txt.untoken 已全部分词至 ../data/jd/middle/train_src.txt
token_file分词 D:\code\data\jd\middle\train_tgt.txt.untoken 将写入 D:\code\data\jd\middle\train_tgt.txt
第一条 白色的，电不足时是发黄
第 100000 行 问客服，客服都会说的
 ---> 问 客 服 ， 客 服 都 会 说 的 
已统计词频 8830
第 200000 行 不大！调到最大声用不了多久，就会卡音！
 ---> 不 大 ！ 调 到 最 大 声 用 不 了 多 久 ， 就 会 卡 音 ！ 
已统计词频 9474
第 300000 行 没良品铺子的好吃
 ---> 没 良 品 铺 子 的 好 吃 
已统计词频 10075
../data/jd/middle/train_tgt.txt.untoken 已全部分词至 ../data/jd/middle/train_tgt.txt
token_file分词 D:\code\data\jd\middle\train_attr.txt.untoken 将写入 D:\code\data\jd\middle\train_attr.txt
../data/jd/middle/train_attr.txt.untoken 不计入词频
第一条 好视力 led充电台灯 护眼学习工作台灯3档调光调色护眼灯TG159TS-WH台灯灯饰照明家装建材
第 100000 行 leohan短裤男原创夏季休闲日系棉麻哈伦七分裤男士薄款7分裤宽松百搭大码运动裤潮短裤男装服饰内衣
 ---> leohan 短 裤 男 原 创 夏 季 休 闲 日 系 棉 麻 哈 伦 七 分 裤 男 士 薄 款 7 分 裤 宽 松 百 搭 大 码 运 动 裤 潮 短 裤 男 装 服 饰 内 衣 
第 200000 行 先科（sast） n-612无线蓝牙音箱 迷你音响便携插卡手机电脑低音炮音箱音箱/音响影音娱乐数码
 ---> 先 科 （ sast ） n - 612 无 线 蓝 牙 音 箱 迷 你 音 响 便 携 插 卡 手 机 电 脑 低 音 炮 音 箱 音 箱 / 音 响 影 音 娱 乐 数 码 
第 300000 行 贤哥 休闲零食台式烤肠 18g*20包/盒休闲零食休闲食品食品饮料
 ---> 贤 哥 休 闲 零 食 台 式 烤 肠 18 g * 20 包 / 盒 休 闲 零 食 休 闲 食 品 食 品 饮 料 
../data/jd/middle/train_attr.txt.untoken 已全部分词至 ../data/jd/middle/train_attr.txt
词频文件已经写入 ../data/jd/middle/vocab.json ../data/jd/middle/vocab.bin
35.013991832733154 秒完成vocab.py
[Info] 原始词库 = 74491
[Info] 频繁字典大小 = 15394, 最低频数 = 20
[Info] 忽略罕词数 = 46307 爆表词汇数 -25509
[Info] 保存词汇到 D:\code\data\jd\reader.data
0.26628804206848145 秒完成vocab.py
'''
