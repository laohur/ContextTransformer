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
    # mydir = "../data/jd/middle"
    mydir = "../data/jd/pure"
    count_file(mydir)
    main(mydir)
    print(time() - t0, "秒完成vocab.py")

'''
C:\ProgramData\Anaconda3\python.exe "C:\Program Files\JetBrains\PyCharm Professional Edition with Anaconda plugin 2019.1.2\helpers\pydev\pydevconsole.py" --mode=client --port=54353
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\code\\ContextTransformer', 'D:/code/ContextTransformer'])
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.4.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 7.4.0
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/vocab.py', wdir='D:/code/ContextTransformer')
token_file分词 D:\code\data\jd\pure\test_src.txt.untoken 将写入 D:\code\data\jd\pure\test_src.txt
第一条 这个是需要充电么？
../data/jd/pure/test_src.txt.untoken 已全部分词至 ../data/jd/pure/test_src.txt
token_file分词 D:\code\data\jd\pure\test_tgt.txt.untoken 将写入 D:\code\data\jd\pure\test_tgt.txt
第一条 当然要冲呀！
../data/jd/pure/test_tgt.txt.untoken 已全部分词至 ../data/jd/pure/test_tgt.txt
token_file分词 D:\code\data\jd\pure\test_attr.txt.untoken 将写入 D:\code\data\jd\pure\test_attr.txt
../data/jd/pure/test_attr.txt.untoken 不计入词频
第一条 雅诗米 生日礼物女生男生实用送女朋友 毕业创意礼品送老婆爱人闺蜜表白结婚 抖音热门同款维他命榨汁杯 创意礼品礼品礼品箱包
../data/jd/pure/test_attr.txt.untoken 已全部分词至 ../data/jd/pure/test_attr.txt
token_file分词 D:\code\data\jd\pure\valid_src.txt.untoken 将写入 D:\code\data\jd\pure\valid_src.txt
第一条 这个手机黄牛要赚死啊
../data/jd/pure/valid_src.txt.untoken 已全部分词至 ../data/jd/pure/valid_src.txt
token_file分词 D:\code\data\jd\pure\valid_tgt.txt.untoken 将写入 D:\code\data\jd\pure\valid_tgt.txt
第一条 赚多少呢
../data/jd/pure/valid_tgt.txt.untoken 已全部分词至 ../data/jd/pure/valid_tgt.txt
token_file分词 D:\code\data\jd\pure\valid_attr.txt.untoken 将写入 D:\code\data\jd\pure\valid_attr.txt
../data/jd/pure/valid_attr.txt.untoken 不计入词频
第一条 黑鲨游戏手机 6GB+64GB 极夜黑 液冷更快 全网通4G 双卡双待手机手机通讯手机
../data/jd/pure/valid_attr.txt.untoken 已全部分词至 ../data/jd/pure/valid_attr.txt
token_file分词 D:\code\data\jd\pure\train_src.txt.untoken 将写入 D:\code\data\jd\pure\train_src.txt
第一条 假清真？详情页中根本没出现清真标。
第 100000 行 能不能将手机游戏投屏上去啊？
 ---> 能 不 能 将 手 机 游 戏 投 屏 上 去 啊 ？ 
已统计词频 6523
第 200000 行 背后信号切口连处，是塑料还是金属
 ---> 背 后 信 号 切 口 连 处 ， 是 塑 料 还 是 金 属 
已统计词频 8196
../data/jd/pure/train_src.txt.untoken 已全部分词至 ../data/jd/pure/train_src.txt
token_file分词 D:\code\data\jd\pure\train_tgt.txt.untoken 将写入 D:\code\data\jd\pure\train_tgt.txt
第一条 没必要假
第 100000 行 当然可以。
 ---> 当 然 可 以 。 
已统计词频 10224
第 200000 行 肯定是塑料啊，不然怎么传输信号
 ---> 肯 定 是 塑 料 啊 ， 不 然 怎 么 传 输 信 号 
已统计词频 11083
../data/jd/pure/train_tgt.txt.untoken 已全部分词至 ../data/jd/pure/train_tgt.txt
token_file分词 D:\code\data\jd\pure\train_attr.txt.untoken 将写入 D:\code\data\jd\pure\train_attr.txt
../data/jd/pure/train_attr.txt.untoken 不计入词频
第一条 伊利 纯牛奶250ml*24盒牛奶乳品饮料冲调食品饮料
第 100000 行 创维(skyworth) 企鹅极光t2 智能网络电视机顶盒4核16g闪存 高清电视盒子无线wifi网络盒子网络产品电脑、办公
 ---> 创 维 ( skyworth ) 企 鹅 极 光 t 2 智 能 网 络 电 视 机 顶 盒 4 核 16 g 闪 存 高 清 电 视 盒 子 无 线 wifi 网 络 盒 子 网 络 产 品 电 脑 、 办 公 
第 200000 行 魅族 魅蓝 e3 全面屏手机  全网通公开版 6gb+64gb 曜石黑 移动联通电信4g手机 双卡双待手机手机通讯手机
 ---> 魅 族 魅 蓝 e 3 全 面 屏 手 机 全 网 通 公 开 版 6 gb + 64 gb 曜 石 黑 移 动 联 通 电 信 4 g 手 机 双 卡 双 待 手 机 手 机 通 讯 手 机 
../data/jd/pure/train_attr.txt.untoken 已全部分词至 ../data/jd/pure/train_attr.txt
词频文件已经写入 ../data/jd/pure/counter.json ../data/jd/pure/counter.bin
[Info] 原始词库 = 11704
[Info] 频繁字典大小 = 4297, 最低频数 = 9
[Info] 忽略罕词数 = 6717 爆表词汇数 -88296
[Info] 保存词汇到 D:\code\data\jd\pure\reader.json
[Info] 保存词汇到 D:\code\data\jd\pure\reader.data
51.52518820762634 秒完成vocab.py
'''