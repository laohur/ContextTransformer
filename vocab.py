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
    parser.add_argument('-min_word_count', type=int, default=10)
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
    count_file(mydir)
    main(mydir)
    print(time() - t0, "秒完成vocab.py")

'''

C:\ProgramData\Anaconda3\python.exe "C:\Program Files\JetBrains\PyCharm Professional Edition with Anaconda plugin 2019.1.2\helpers\pydev\pydevconsole.py" --mode=client --port=52514
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\code\\ContextTransformer', 'D:/code/ContextTransformer'])
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.4.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 7.4.0
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/vocab.py', wdir='D:/code/ContextTransformer')
token_file分词 D:\code\data\jd\big\test_src.txt.untoken 将写入 D:\code\data\jd\big\test_src.txt
第一条 充电宝自己坏了，不充电了啊。还换吗
../data/jd/big/test_src.txt.untoken 已全部分词至 ../data/jd/big/test_src.txt
token_file分词 D:\code\data\jd\big\test_tgt.txt.untoken 将写入 D:\code\data\jd\big\test_tgt.txt
第一条 不要买
../data/jd/big/test_tgt.txt.untoken 已全部分词至 ../data/jd/big/test_tgt.txt
token_file分词 D:\code\data\jd\big\test_attr.txt.untoken 将写入 D:\code\data\jd\big\test_attr.txt
../data/jd/big/test_attr.txt.untoken 不计入词频
第一条 半岛铁盒 K20精英版 21200毫安移动电源高倍率动力电芯大容量LED手电数字屏显安卓/苹果通用型充电宝 白色移动电源手机配件手机
../data/jd/big/test_attr.txt.untoken 已全部分词至 ../data/jd/big/test_attr.txt
token_file分词 D:\code\data\jd\big\valid_src.txt.untoken 将写入 D:\code\data\jd\big\valid_src.txt
第一条 可以降血糖吗
../data/jd/big/valid_src.txt.untoken 已全部分词至 ../data/jd/big/valid_src.txt
token_file分词 D:\code\data\jd\big\valid_tgt.txt.untoken 将写入 D:\code\data\jd\big\valid_tgt.txt
第一条 似乎没有效果，，，，，，，，，，。
../data/jd/big/valid_tgt.txt.untoken 已全部分词至 ../data/jd/big/valid_tgt.txt
token_file分词 D:\code\data\jd\big\valid_attr.txt.untoken 将写入 D:\code\data\jd\big\valid_attr.txt
../data/jd/big/valid_attr.txt.untoken 不计入词频
第一条 汤臣倍健（BY-HEALTH） 铬酵母片90片维生素/矿物质营养成分医药保健
../data/jd/big/valid_attr.txt.untoken 已全部分词至 ../data/jd/big/valid_attr.txt
token_file分词 D:\code\data\jd\big\train_src.txt.untoken 将写入 D:\code\data\jd\big\train_src.txt
第一条 这个产品可以帮助鱼消化吗？
第 100000 行 这双鞋的鞋底软吗？
 ---> 这 双 鞋 的 鞋 底 软 吗 ？ 
已统计词频 6495
第 200000 行 请问骑过新日翼虎这款车的车主，72v20a的那款，大概能骑多少公里，质量怎么样
 ---> 请 问 骑 过 新 日 翼 虎 这 款 车 的 车 主 ， 72 v 20 a 的 那 款 ， 大 概 能 骑 多 少 公 里 ， 质 量 怎 么 样 
已统计词频 8137
第 300000 行 新百伦的鞋
 ---> 新 百 伦 的 鞋 
已统计词频 9411
第 400000 行 我的冰箱内壁为什么会结冰？
 ---> 我 的 冰 箱 内 壁 为 什 么 会 结 冰 ？ 
已统计词频 10420
第 500000 行 内胆什么材质的？
 ---> 内 胆 什 么 材 质 的 ？ 
已统计词频 11283
第 600000 行 那个返回键会不好用吗？
 ---> 那 个 返 回 键 会 不 好 用 吗 ？ 
已统计词频 12120
第 700000 行 多长的桌子能摆的下啊
 ---> 多 长 的 桌 子 能 摆 的 下 啊 
已统计词频 12794
第 800000 行 不打孔会不会不牢固啊？
 ---> 不 打 孔 会 不 会 不 牢 固 啊 ？ 
已统计词频 13475
第 900000 行 充满电会自动断电吗
 ---> 充 满 电 会 自 动 断 电 吗 
已统计词频 14086
第 1000000 行 买香肠送蒸锅吗？
 ---> 买 香 肠 送 蒸 锅 吗 ？ 
已统计词频 14669
第 1100000 行 风量大么？
 ---> 风 量 大 么 ？ 
已统计词频 15225
第 1200000 行 114斤165cm穿多大码合适
 ---> 114 斤 165 cm 穿 多 大 码 合 适 
已统计词频 15732
第 1300000 行 买这个有什么送的，保养怎么样
 ---> 买 这 个 有 什 么 送 的 ， 保 养 怎 么 样 
已统计词频 16160
第 1400000 行 我爸一动就气急能吃吗？
 ---> 我 爸 一 动 就 气 急 能 吃 吗 ？ 
已统计词频 16620
第 1500000 行 是银的吗？
 ---> 是 银 的 吗 ？ 
已统计词频 17108
第 1600000 行 这款冰箱会不会有水流声
 ---> 这 款 冰 箱 会 不 会 有 水 流 声 
已统计词频 17542
../data/jd/big/train_src.txt.untoken 已全部分词至 ../data/jd/big/train_src.txt
token_file分词 D:\code\data\jd\big\train_tgt.txt.untoken 将写入 D:\code\data\jd\big\train_tgt.txt
第一条 不可以帮助鱼的消化，是控制水质的可以分解鱼的粪便
第 100000 行 嗯额！！
 ---> 嗯 额 ！ ！ 
已统计词频 18429
第 200000 行 45
 ---> 45 
已统计词频 18843
第 300000 行 什么意思
 ---> 什 么 意 思 
已统计词频 19268
第 400000 行 我的也会。只要东西不靠着，它就会流走。
 ---> 我 的 也 会 。 只 要 东 西 不 靠 着 ， 它 就 会 流 走 。 
已统计词频 19634
第 500000 行 这锅的电脑板不好用其他地方还没那董内胆非常好我喜欢蒸出饭来好吃
 ---> 这 锅 的 电 脑 板 不 好 用 其 他 地 方 还 没 那 董 内 胆 非 常 好 我 喜 欢 蒸 出 饭 来 好 吃 
已统计词频 20003
第 600000 行 这就尴尬了，个人习惯吧
 ---> 这 就 尴 尬 了 ， 个 人 习 惯 吧 
已统计词频 20374
第 700000 行 还好吧
 ---> 还 好 吧 
已统计词频 20724
第 800000 行 不知道，我忘记有贴的了，已经打孔了
 ---> 不 知 道 ， 我 忘 记 有 贴 的 了 ， 已 经 打 孔 了 
已统计词频 21053
第 900000 行 充电红灯，充满灯灭，应该是自动断电了
 ---> 充 电 红 灯 ， 充 满 灯 灭 ， 应 该 是 自 动 断 电 了 
已统计词频 21418
第 1000000 行 不送的
 ---> 不 送 的 
已统计词频 21746
第 1100000 行 大
 ---> 大 
已统计词频 22042
第 1200000 行 m吧，问问客服
 ---> m 吧 ， 问 问 客 服 
已统计词频 22353
第 1300000 行 什么也没送，挺好的
 ---> 什 么 也 没 送 ， 挺 好 的 
已统计词频 22661
第 1400000 行 能
 ---> 能 
已统计词频 22932
第 1500000 行 是的
 ---> 是 的 
已统计词频 23234
第 1600000 行 冰箱在运行中，出现的轻微呼噜声、气泡声、流水声，这些声音是属于制冷剂在管路内流动或蒸发时发出的正常声音，只要冰箱通电运行，这类声音会一直间断的存在，是正常的哦，请放心使用！
 ---> 冰 箱 在 运 行 中 ， 出 现 的 轻 微 呼 噜 声 、 气 泡 声 、 流 水 声 ， 这 些 声 音 是 属 于 制 冷 剂 在 管 路 内 流 动 或 蒸 发 时 发 出 的 正 常 声 音 ， 只 要 冰 箱 通 电 运 行 ， 这 类 声 音 会 一 直 间 断 的 存 在 ， 是 正 常 的 哦 ， 请 放 心 使 用 ！ 
已统计词频 23578
../data/jd/big/train_tgt.txt.untoken 已全部分词至 ../data/jd/big/train_tgt.txt
token_file分词 D:\code\data\jd\big\train_attr.txt.untoken 将写入 D:\code\data\jd\big\train_attr.txt
../data/jd/big/train_attr.txt.untoken 不计入词频
第一条 硝化细菌水鱼缸水质浓缩硝化菌（非鱼药）消化细菌净水剂消毒水族药剂水族宠物生活
第 100000 行 荣仕健康鞋 一脚蹬女鞋透气网面妈妈鞋轻便老人运动鞋平底休闲老北京布鞋防滑软底中老年健步鞋妈妈鞋时尚女鞋鞋靴
 ---> 荣 仕 健 康 鞋 一 脚 蹬 女 鞋 透 气 网 面 妈 妈 鞋 轻 便 老 人 运 动 鞋 平 底 休 闲 老 北 京 布 鞋 防 滑 软 底 中 老 年 健 步 鞋 妈 妈 鞋 时 尚 女 鞋 鞋 靴 
第 200000 行 新日（sunra） 电动车电瓶车豪华运动款男士款踏板车成人越野电动摩托车电动车骑行运动运动户外
 ---> 新 日 （ sunra ） 电 动 车 电 瓶 车 豪 华 运 动 款 男 士 款 踏 板 车 成 人 越 野 电 动 摩 托 车 电 动 车 骑 行 运 动 运 动 户 外 
第 300000 行 皇宇 绒皮护理清洁剂180ml套装（无色护理剂+去污剂+绒皮刷）皮具护理品皮具护理家庭清洁/纸品
 ---> 皇 宇 绒 皮 护 理 清 洁 剂 180 ml 套 装 （ 无 色 护 理 剂 + 去 污 剂 + 绒 皮 刷 ） 皮 具 护 理 品 皮 具 护 理 家 庭 清 洁 / 纸 品 
第 400000 行 康佳（konka）288升 静音保鲜 多门冰箱 分类存储（银色）bcd-288gy4s冰箱大 家 电家用电器
 ---> 康 佳 （ konka ） 288 升 静 音 保 鲜 多 门 冰 箱 分 类 存 储 （ 银 色 ） bcd - 288 gy 4 s 冰 箱 大 家 电 家 用 电 器 
第 500000 行 小米生活 电饭煲迷你预约智能多功能1.2l饭锅电饭煲厨房小电家用电器
 ---> 小 米 生 活 电 饭 煲 迷 你 预 约 智 能 多 功 能 1 . 2 l 饭 锅 电 饭 煲 厨 房 小 电 家 用 电 器 
第 600000 行 小米（mi） 小米 note3 手机手机手机通讯手机
 ---> 小 米 （ mi ） 小 米 note 3 手 机 手 机 手 机 通 讯 手 机 
第 700000 行 三星（samsung）ua55muf30zjxxz 55英寸 4k超高清 智能网络 液晶平板电视  黑色平板电视大 家 电家用电器
 ---> 三 星 （ samsung ） ua 55 muf 30 zjxxz 55 英 寸 4 k 超 高 清 智 能 网 络 液 晶 平 板 电 视 黑 色 平 板 电 视 大 家 电 家 用 电 器 
第 800000 行 金佰福 【大冲力】 冲马桶冲厕所冲水箱 蹲坑蹲便器冲便器冲水箱 静音水箱双按节能大冲力水箱马桶厨房卫浴家装建材
 ---> 金 佰 福 【 大 冲 力 】 冲 马 桶 冲 厕 所 冲 水 箱 蹲 坑 蹲 便 器 冲 便 器 冲 水 箱 静 音 水 箱 双 按 节 能 大 冲 力 水 箱 马 桶 厨 房 卫 浴 家 装 建 材 
第 900000 行 1more万魔 ibfree蓝牙耳机 运动耳机 黑色耳机/耳麦影音娱乐数码
 ---> 1 more 万 魔 ibfree 蓝 牙 耳 机 运 动 耳 机 黑 色 耳 机 / 耳 麦 影 音 娱 乐 数 码 
第 1000000 行 皇上皇 中华老字号 合家欢腊肠（5分瘦）400g熟食腊味休闲食品食品饮料
 ---> 皇 上 皇 中 华 老 字 号 合 家 欢 腊 肠 （ 5 分 瘦 ） 400 g 熟 食 腊 味 休 闲 食 品 食 品 饮 料 
第 1100000 行 京天（kotin） 鑫谷光韵12cm炫酷光圈/减震/静音/台式组装电脑主机机箱散热风扇散热器电脑配件电脑、办公
 ---> 京 天 （ kotin ） 鑫 谷 光 韵 12 cm 炫 酷 光 圈 / 减 震 / 静 音 / 台 式 组 装 电 脑 主 机 机 箱 散 热 风 扇 散 热 器 电 脑 配 件 电 脑 、 办 公 
第 1200000 行 法格娣 连衣裙两件套装2018春季新款韩版宽松衬衫加中长a字连衣裙女装服饰内衣
 ---> 法 格 娣 连 衣 裙 两 件 套 装 2018 春 季 新 款 韩 版 宽 松 衬 衫 加 中 长 a 字 连 衣 裙 女 装 服 饰 内 衣 
第 1300000 行 小米（mi）小米电视4a 50英寸 l50m5-ad 2gb+8gb hdr 4k超高清 蓝牙语音遥控 人工智能语音网络液晶平板电视平板电视大 家 电家用电器
 ---> 小 米 （ mi ） 小 米 电 视 4 a 50 英 寸 l 50 m 5 - ad 2 gb + 8 gb hdr 4 k 超 高 清 蓝 牙 语 音 遥 控 人 工 智 能 语 音 网 络 液 晶 平 板 电 视 平 板 电 视 大 家 电 家 用 电 器 
第 1400000 行 养无极 补肺丸 9g*10丸*4板（气短喘咳 干咳痰粘 咽干）感冒咳嗽中西药品医药保健
 ---> 养 无 极 补 肺 丸 9 g * 10 丸 * 4 板 （ 气 短 喘 咳 干 咳 痰 粘 咽 干 ） 感 冒 咳 嗽 中 西 药 品 医 药 保 健 
第 1500000 行 s925纯银豆豆耳钉男女 时尚简约养耳圆珠耳针 日韩百搭耳环耳棒耳饰时尚饰品珠宝首饰
 ---> s 925 纯 银 豆 豆 耳 钉 男 女 时 尚 简 约 养 耳 圆 珠 耳 针 日 韩 百 搭 耳 环 耳 棒 耳 饰 时 尚 饰 品 珠 宝 首 饰 
第 1600000 行 美的(midea)655升 对开门冰箱 变频无霜 一级能效 智能app 大容量电冰箱 米兰金 bcd-655wkpzm(e)冰箱大 家 电家用电器
 ---> 美 的 ( midea ) 655 升 对 开 门 冰 箱 变 频 无 霜 一 级 能 效 智 能 app 大 容 量 电 冰 箱 米 兰 金 bcd - 655 wkpzm ( e ) 冰 箱 大 家 电 家 用 电 器 
../data/jd/big/train_attr.txt.untoken 已全部分词至 ../data/jd/big/train_attr.txt
词频文件已经写入 ../data/jd/big/counter.json ../data/jd/big/counter.bin
[Info] 原始词库 = 23836
[Info] 频繁字典大小 = 6958, 最低频数 = 10
[Info] 忽略罕词数 = 14270 爆表词汇数 -76164
[Info] 保存词汇到 D:\code\data\jd\big\reader.json
[Info] 保存词汇到 D:\code\data\jd\big\reader.data
215.18439364433289 秒完成vocab.py




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
