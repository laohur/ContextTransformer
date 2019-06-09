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
    parser.add_argument('-min_word_count', type=int, default=5)
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
    mydir = "../data/jd/big"
    # count_file(mydir)
    main(mydir)
    print(time() - t0, "秒完成vocab.py")

'''
C:\ProgramData\Anaconda3\python.exe "C:\Program Files\JetBrains\PyCharm Professional Edition with Anaconda plugin 2019.1.2\helpers\pydev\pydevconsole.py" --mode=client --port=53146
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\code\\ContextTransformer', 'D:/code/ContextTransformer'])
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.4.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 7.4.0
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/vocab.py', wdir='D:/code/ContextTransformer')
token_file分词 D:\code\data\jd\big\test_src.txt.untoken 将写入 D:\code\data\jd\big\test_src.txt
第一条 玩游戏dnf怎么样
../data/jd/big/test_src.txt.untoken 已全部分词至 ../data/jd/big/test_src.txt
token_file分词 D:\code\data\jd\big\test_tgt.txt.untoken 将写入 D:\code\data\jd\big\test_tgt.txt
第一条 没问题
../data/jd/big/test_tgt.txt.untoken 已全部分词至 ../data/jd/big/test_tgt.txt
token_file分词 D:\code\data\jd\big\test_attr.txt.untoken 将写入 D:\code\data\jd\big\test_attr.txt
../data/jd/big/test_attr.txt.untoken 不计入词频
第一条 铭影GTX750Ti 2G显卡战神 吃鸡游戏显卡 台式机电脑显卡750系列显卡2g独立显卡显卡电脑配件电脑、办公
../data/jd/big/test_attr.txt.untoken 已全部分词至 ../data/jd/big/test_attr.txt
token_file分词 D:\code\data\jd\big\valid_src.txt.untoken 将写入 D:\code\data\jd\big\valid_src.txt
第一条 这款可以同时连几部手机？连接不用配对密码的吗？
../data/jd/big/valid_src.txt.untoken 已全部分词至 ../data/jd/big/valid_src.txt
token_file分词 D:\code\data\jd\big\valid_tgt.txt.untoken 将写入 D:\code\data\jd\big\valid_tgt.txt
第一条 密码到不用，就是蓝牙信号不好。
../data/jd/big/valid_tgt.txt.untoken 已全部分词至 ../data/jd/big/valid_tgt.txt
token_file分词 D:\code\data\jd\big\valid_attr.txt.untoken 将写入 D:\code\data\jd\big\valid_attr.txt
../data/jd/big/valid_attr.txt.untoken 不计入词频
第一条 三星（SAMSUNG）Level U 项圈式 运动蓝牙音乐耳机（雅墨黑）蓝牙耳机手机配件手机
../data/jd/big/valid_attr.txt.untoken 已全部分词至 ../data/jd/big/valid_attr.txt
token_file分词 D:\code\data\jd\big\train_src.txt.untoken 将写入 D:\code\data\jd\big\train_src.txt
第一条 请问标准版和青春版差别在哪儿啊
第 100000 行 语音遥控器可以直接按键换台吗
 ---> 语 音 遥 控 器 可 以 直 接 按 键 换 台 吗 
已统计词频 6606
第 200000 行 我的充电宝充不进去电怎么回事？
 ---> 我 的 充 电 宝 充 不 进 去 电 怎 么 回 事 ？ 
已统计词频 8255
第 300000 行 笔记本内录时混响可以解决吗？
 ---> 笔 记 本 内 录 时 混 响 可 以 解 决 吗 ？ 
已统计词频 9469
第 400000 行 怎么看屏幕是哪家产的
 ---> 怎 么 看 屏 幕 是 哪 家 产 的 
已统计词频 10496
第 500000 行 刚买回来c盘就占用了49g内存，可以释放一下吗
 ---> 刚 买 回 来 c 盘 就 占 用 了 49 g 内 存 ， 可 以 释 放 一 下 吗 
已统计词频 11394
第 600000 行 这款本子用cad，pkpm，天正，广联达流畅吗
 ---> 这 款 本 子 用 cad ， pkpm ， 天 正 ， 广 联 达 流 畅 吗 
已统计词频 12141
../data/jd/big/train_src.txt.untoken 已全部分词至 ../data/jd/big/train_src.txt
token_file分词 D:\code\data\jd\big\train_tgt.txt.untoken 将写入 D:\code\data\jd\big\train_tgt.txt
第一条 屏幕一个45一个75，和键盘灯，一个RGB，一个只有白光
第 100000 行 可以
 ---> 可 以 
已统计词频 13183
第 200000 行 找售后啊。
 ---> 找 售 后 啊 。 
已统计词频 13836
第 300000 行 不能，说是声卡，其实就是转接头。
 ---> 不 能 ， 说 是 声 卡 ， 其 实 就 是 转 接 头 。 
已统计词频 14439
第 400000 行 奇美
 ---> 奇 美 
已统计词频 14961
第 500000 行 可以
 ---> 可 以 
已统计词频 15437
第 600000 行 cad什么都可以，完全流程的，3dmx都可以
 ---> cad 什 么 都 可 以 ， 完 全 流 程 的 ， 3 dmx 都 可 以 
已统计词频 15875
../data/jd/big/train_tgt.txt.untoken 已全部分词至 ../data/jd/big/train_tgt.txt
token_file分词 D:\code\data\jd\big\train_attr.txt.untoken 将写入 D:\code\data\jd\big\train_attr.txt
../data/jd/big/train_attr.txt.untoken 不计入词频
第一条 炫龙（Shinelon）毁灭者KP2 GTX1060  6G独显 15.6英寸游戏笔记本电脑（i5-8400 8G 128G+1TB IPS）游戏本电脑整机电脑、办公
第 100000 行 小米（mi）小米电视4a 50英寸 l50m5-ad 2gb+8gb hdr 4k超高清 蓝牙语音遥控 人工智能语音网络液晶平板电视平板电视大 家 电家用电器
 ---> 小 米 （ mi ） 小 米 电 视 4 a 50 英 寸 l 50 m 5 - ad 2 gb + 8 gb hdr 4 k 超 高 清 蓝 牙 语 音 遥 控 人 工 智 能 语 音 网 络 液 晶 平 板 电 视 平 板 电 视 大 家 电 家 用 电 器 
第 200000 行 飞毛腿f20 type-c/micro双输入大屏电量显示 聚合物移动电源/充电宝 20000毫安黑色适用于苹果/三星/华为/小米移动电源手机配件手机
 ---> 飞 毛 腿 f 20 type - c / micro 双 输 入 大 屏 电 量 显 示 聚 合 物 移 动 电 源 / 充 电 宝 20000 毫 安 黑 色 适 用 于 苹 果 / 三 星 / 华 为 / 小 米 移 动 电 源 手 机 配 件 手 机 
第 300000 行 绿联（ugreen）usb外置独立声卡 免驱台式机笔记本电脑转3.5mm立体音频接口转换器麦克风耳机音响转接头30712线缆外设产品电脑、办公
 ---> 绿 联 （ ugreen ） usb 外 置 独 立 声 卡 免 驱 台 式 机 笔 记 本 电 脑 转 3 . 5 mm 立 体 音 频 接 口 转 换 器 麦 克 风 耳 机 音 响 转 接 头 30712 线 缆 外 设 产 品 电 脑 、 办 公 
第 400000 行 联想(lenovo)拯救者r720 15.6英寸游戏笔记本电脑(i7-7700hq 8g 1t+128g ssd gtx1050 2g ips)黑游戏本电脑整机电脑、办公
 ---> 联 想 ( lenovo ) 拯 救 者 r 720 15 . 6 英 寸 游 戏 笔 记 本 电 脑 ( i 7 - 7700 hq 8 g 1 t + 128 g ssd gtx 1050 2 g ips ) 黑 游 戏 本 电 脑 整 机 电 脑 、 办 公 
第 500000 行 戴尔dell灵越游匣gtx1050 15.6英寸游戏笔记本电脑(i5-7300hq 8g 128gssd+1t 4g独显 ips 散热快)黑游戏本电脑整机电脑、办公
 ---> 戴 尔 dell 灵 越 游 匣 gtx 1050 15 . 6 英 寸 游 戏 笔 记 本 电 脑 ( i 5 - 7300 hq 8 g 128 gssd + 1 t 4 g 独 显 ips 散 热 快 ) 黑 游 戏 本 电 脑 整 机 电 脑 、 办 公 
第 600000 行 惠普(hp)暗影精灵2代pro 15.6英寸游戏笔记本电脑(i7-7700hq 8g 128g pcie ssd+1t gtx1050 2g独显 ips)游戏本电脑整机电脑、办公
 ---> 惠 普 ( hp ) 暗 影 精 灵 2 代 pro 15 . 6 英 寸 游 戏 笔 记 本 电 脑 ( i 7 - 7700 hq 8 g 128 g pcie ssd + 1 t gtx 1050 2 g 独 显 ips ) 游 戏 本 电 脑 整 机 电 脑 、 办 公 
../data/jd/big/train_attr.txt.untoken 已全部分词至 ../data/jd/big/train_attr.txt
词频文件已经写入 ../data/jd/big/counter.json ../data/jd/big/counter.bin
[Info] 原始词库 = 15967
[Info] 频繁字典大小 = 5218, 最低频数 = 10
[Info] 忽略罕词数 = 9393 爆表词汇数 -84033
[Info] 保存词汇到 D:\code\data\jd\big\reader.json
[Info] 保存词汇到 D:\code\data\jd\big\reader.data
89.24953603744507 秒完成vocab.py

[Info] 原始词库 = 15967
[Info] 频繁字典大小 = 6558, 最低频数 = 5
[Info] 忽略罕词数 = 8053 爆表词汇数 -84033
[Info] 保存词汇到 D:\code\data\jd\big\reader.json
[Info] 保存词汇到 D:\code\data\jd\big\reader.data
0.13234996795654297 秒完成vocab.py


'''
