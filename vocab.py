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


def token_file(dir, source, target, begin=0, end=sys.maxsize, keep_case=False):
    if end < 0:
        end = sys.maxsize
    print("token_file分词", os.path.abspath(source), "将写入", os.path.abspath(target))
    counter = {}
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

        # 统计词频,第一段商品编码应该去除
        for token in tokens[2:]:
            if token in counter:
                counter[token] += 1
            else:
                counter[token] = 1

        tokens = " ".join(tokens)  # 自带"\n"
        f2.write(tokens)
        if i % 100000 == 0:
            print("第", i, "行", line, "--->", tokens)
            print("已统计词频", len(counter))
        line = f1.readline()
    f1.close()
    f2.close()
    print(source, "已全部分词至", target)
    counter = dict(sorted(counter.items(), key=lambda kv: kv[1], reverse=True))

    with open(dir + "/counter.json", "w", encoding="utf-8") as f:  # 特殊字符有问题，仅供人类阅读
        json.dump(counter, f, ensure_ascii=False)
    torch.save(counter, dir + "/counter.bin")
    print("词频文件已经写入", dir + "/vocab.json", dir + "/vocab.bin")


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
    dir = "../data/jd/big"

    dir = "../data/jd"

    source = dir + "/full.skuqa"
    target = "C:/Intel" + "/tokenized.skuqa"  # 最好分开磁盘，速度没啥变化，3.5m/s
    token_file(dir=dir, source=source, target=target)


def main():
    dir = "../data/jd"

    parser = argparse.ArgumentParser()
    parser.add_argument('-save_dir', default=dir)
    parser.add_argument('-counter_path', default=dir + "/counter.bin")
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=20)
    parser.add_argument('-min_word_count', type=int, default=20)
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

    path = args.save_dir + "/reader.data"
    print('[Info] 保存词汇到', os.path.abspath(path))
    torch.save(vocab, path)


if __name__ == '__main__':
    t0 = time()
    # count_file()
    main()
    print(time() - t0, "秒完成vocab.py")

'''
token_file分词 D:\code\data\jd\full.skuqa 将写入 C:\Intel\tokenized.skuqa
第一条 skuid 	 attributes 	 review  	 counter 	 question 	 answer
第 100000 行 1077731041890	【滨海馆】虾酱麻虾酱拌面酱下饭菜海鲜酱盐城特产蜢子虾酱170g/瓶调味品粮油调味食品饮料	很喜欢哦～～很不错的东西～	1	咸不咸？我记得应该不咸吧	不是很闲，正好吧。
 ---> 1077731041890 	 【 滨 海 馆 】 虾 酱 麻 虾 酱 拌 面 酱 下 饭 菜 海 鲜 酱 盐 城 特 产 蜢 子 虾 酱 170 g / 瓶 调 味 品 粮 油 调 味 食 品 饮 料 	 很 喜 欢 哦 ～ ～ 很 不 错 的 东 西 ～ 	 1 	 咸 不 咸 ？ 我 记 得 应 该 不 咸 吧 	 不 是 很 闲 ， 正 好 吧 。 
第 200000 行 169458848	技嘉(gigabyte)geforce gtx 1050ti oc 1316-1430mhz/7008mhz 4g/128bit游戏显卡显卡电脑配件电脑、办公	很烦心的个显卡，用了几天就坏了，退不了，又换不了，烦啊	5	1050ti吃鸡靠谱吗？现在一斯雾九，这个车能开吗	能，最好1060
 ---> 169458848 	 技 嘉 ( gigabyte ) geforce gtx 1050 ti oc 1316 - 1430 mhz / 7008 mhz 4 g / 128 bit 游 戏 显 卡 显 卡 电 脑 配 件 电 脑 、 办 公 	 很 烦 心 的 个 显 卡 ， 用 了 几 天 就 坏 了 ， 退 不 了 ， 又 换 不 了 ， 烦 啊 	 5 	 1050 ti 吃 鸡 靠 谱 吗 ？ 现 在 一 斯 雾 九 ， 这 个 车 能 开 吗 	 能 ， 最 好 1060 
第 300000 行 1217372566652	南极人 短袖t恤女2018夏季新款韩版宽松大码女士打底上衣字母印花时尚体恤 as062t恤女装服饰内衣	物流不错，物流速度挺快的，很好的一次购物	17	是棉的吗？	是棉的，上身可舒服了，喜欢
 ---> 1217372566652 	 南 极 人 短 袖 t 恤 女 2018 夏 季 新 款 韩 版 宽 松 大 码 女 士 打 底 上 衣 字 母 印 花 时 尚 体 恤 as 062 t 恤 女 装 服 饰 内 衣 	 物 流 不 错 ， 物 流 速 度 挺 快 的 ， 很 好 的 一 次 购 物 	 17 	 是 棉 的 吗 ？ 	 是 棉 的 ， 上 身 可 舒 服 了 ， 喜 欢 
第 400000 行 1214629215363	时域新款假两件弹力速干紧身运动跑步瑜伽黑桃心健身房长裤女春秋运动裤运动服饰运动户外	这个衣服穿着美美哒，运动也方便，好喜欢	1	这个会不会掉色啊？	蛮好的，没有这个现象，有点小性感
 ---> 1214629215363 	 时 域 新 款 假 两 件 弹 力 速 干 紧 身 运 动 跑 步 瑜 伽 黑 桃 心 健 身 房 长 裤 女 春 秋 运 动 裤 运 动 服 饰 运 动 户 外 	 这 个 衣 服 穿 着 美 美 哒 ， 运 动 也 方 便 ， 好 喜 欢 	 1 	 这 个 会 不 会 掉 色 啊 ？ 	 蛮 好 的 ， 没 有 这 个 现 象 ， 有 点 小 性 感 
第 500000 行 491638750836	乐视超级电视 超4 x50 pro 50英寸 rgb真4k超高清液晶3d屏幕（标配挂架）平板电视大 家 电家用电器	电视很好 反应快  功能多 内存也大 值了	4	能去除开机广告吗	不能
 ---> 491638750836 	 乐 视 超 级 电 视 超 4 x 50 pro 50 英 寸 rgb 真 4 k 超 高 清 液 晶 3 d 屏 幕 （ 标 配 挂 架 ） 平 板 电 视 大 家 电 家 用 电 器 	 电 视 很 好 反 应 快 功 能 多 内 存 也 大 值 了 	 4 	 能 去 除 开 机 广 告 吗 	 不 能 
第 600000 行 477021908192	一杯香茶叶 2018新茶明前龙井茶 绿茶茶叶买1送1共200克装茶叶明前绿茶散装浓香礼盒装龙井茗茶食品饮料	挺好的，喝完还会回购的！	173	有买过新上市的茶的朋友吗？是要鲜嫩好喝一点吗？	好
 ---> 477021908192 	 一 杯 香 茶 叶 2018 新 茶 明 前 龙 井 茶 绿 茶 茶 叶 买 1 送 1 共 200 克 装 茶 叶 明 前 绿 茶 散 装 浓 香 礼 盒 装 龙 井 茗 茶 食 品 饮 料 	 挺 好 的 ， 喝 完 还 会 回 购 的 ！ 	 173 	 有 买 过 新 上 市 的 茶 的 朋 友 吗 ？ 是 要 鲜 嫩 好 喝 一 点 吗 ？ 	 好 
第 700000 行 280009731	鬼塚虎（onitsuka tiger）中性款  休闲鞋 serrano    d109l-0142 白色休闲鞋运动鞋包运动户外	一般般	1	男女同款吗	对，中性款，男女一样
 ---> 280009731 	 鬼 塚 虎 （ onitsuka tiger ） 中 性 款 休 闲 鞋 serrano d 109 l - 0142 白 色 休 闲 鞋 运 动 鞋 包 运 动 户 外 	 一 般 般 	 1 	 男 女 同 款 吗 	 对 ， 中 性 款 ， 男 女 一 样 
第 800000 行 236454771	金士顿（kingston）dt se9h 32gb u盘 个性化 自定义定制 金属车载u盘u盘外设产品电脑、办公	很好，物流很快，专业刻字很好	11	可以刻字吗	可以刻字
 ---> 236454771 	 金 士 顿 （ kingston ） dt se 9 h 32 gb u 盘 个 性 化 自 定 义 定 制 金 属 车 载 u 盘 u 盘 外 设 产 品 电 脑 、 办 公 	 很 好 ， 物 流 很 快 ， 专 业 刻 字 很 好 	 11 	 可 以 刻 字 吗 	 可 以 刻 字 
第 900000 行 220509135	apple ipad 平板电脑 9.7英寸（128g wlan版/a9 芯片/retina显示屏/touch id技术 mp2h2ch/a）深空灰色平板电脑电脑整机电脑、办公	作为入门级的备用全尺寸ipad，ipad（2017）不失为一个经济实惠的选择，电池续航能力高，retina屏幕显示清晰，虽然不如全贴合屏幕效果那么艳丽，但平时看个视频读个pdf还是足够的，对于不是重度依赖使用者还是比较实用的，推荐~	107	苹果品牌日会便宜很多吗	便宜一点点，不会超过200
 ---> 220509135 	 apple ipad 平 板 电 脑 9 . 7 英 寸 （ 128 g wlan 版 / a 9 芯 片 / retina 显 示 屏 / touch id 技 术 mp 2 h 2 ch / a ） 深 空 灰 色 平 板 电 脑 电 脑 整 机 电 脑 、 办 公 	 作 为 入 门 级 的 备 用 全 尺 寸 ipad ， ipad （ 2017 ） 不 失 为 一 个 经 济 实 惠 的 选 择 ， 电 池 续 航 能 力 高 ， retina 屏 幕 显 示 清 晰 ， 虽 然 不 如 全 贴 合 屏 幕 效 果 那 么 艳 丽 ， 但 平 时 看 个 视 频 读 个 pdf 还 是 足 够 的 ， 对 于 不 是 重 度 依 赖 使 用 者 还 是 比 较 实 用 的 ， 推 荐 ~ 	 107 	 苹 果 品 牌 日 会 便 宜 很 多 吗 	 便 宜 一 点 点 ， 不 会 超 过 200 
第 1000000 行 70305717516	下单立减25 英国mag羊奶粉 幼犬成犬狗狗羊奶粉猫咪幼猫成猫羊奶粉益生菌配方代母乳400g奶粉医疗保健宠物生活	不错	9	半岁萨摩，喝怎么样。要给多少量，多少水	半岁有点大啦，而且羊奶粉要少喝，上火的东西
 ---> 70305717516 	 下 单 立 减 25 英 国 mag 羊 奶 粉 幼 犬 成 犬 狗 狗 羊 奶 粉 猫 咪 幼 猫 成 猫 羊 奶 粉 益 生 菌 配 方 代 母 乳 400 g 奶 粉 医 疗 保 健 宠 物 生 活 	 不 错 	 9 	 半 岁 萨 摩 ， 喝 怎 么 样 。 要 给 多 少 量 ， 多 少 水 	 半 岁 有 点 大 啦 ， 而 且 羊 奶 粉 要 少 喝 ， 上 火 的 东 西 
第 1100000 行 284521531	小米(mi) air 12.5英寸金属超轻薄笔记本电脑(core m-7y30 4g 128g 全高清屏 背光键盘 win10正版office)银笔记本电脑整机电脑、办公	没送什么配件，电脑还不错，物有所值	325	是不是不连网就不会被激活？在联网的状态下是自动激活还是要自己操作的？这激活指的是window系统吧？我是怕莫名被激活没法无理由退货了。	联网自动激活
 ---> 284521531 	 小 米 ( mi ) air 12 . 5 英 寸 金 属 超 轻 薄 笔 记 本 电 脑 ( core m - 7 y 30 4 g 128 g 全 高 清 屏 背 光 键 盘 win 10 正 版 office ) 银 笔 记 本 电 脑 整 机 电 脑 、 办 公 	 没 送 什 么 配 件 ， 电 脑 还 不 错 ， 物 有 所 值 	 325 	 是 不 是 不 连 网 就 不 会 被 激 活 ？ 在 联 网 的 状 态 下 是 自 动 激 活 还 是 要 自 己 操 作 的 ？ 这 激 活 指 的 是 window 系 统 吧 ？ 我 是 怕 莫 名 被 激 活 没 法 无 理 由 退 货 了 。 	 联 网 自 动 激 活 
第 1200000 行 301381771	oppo a73 高配版 全面屏拍照手机 4gb+64gb 黑色 全网通 移动联通电信4g 双卡双待手机手机手机通讯手机	搞不懂一个手机手电筒设计那么麻烦，还要在设置里找，真佩服opp0狗屎设计理念😡😡😡😡	97	手机内置系统多吗？	不多，
 ---> 301381771 	 oppo a 73 高 配 版 全 面 屏 拍 照 手 机 4 gb + 64 gb 黑 色 全 网 通 移 动 联 通 电 信 4 g 双 卡 双 待 手 机 手 机 手 机 通 讯 手 机 	 搞 不 懂 一 个 手 机 手 电 筒 设 计 那 么 麻 烦 ， 还 要 在 设 置 里 找 ， 真 佩 服 opp 0 狗 屎 设 计 理 念 😡 😡 😡 😡 	 97 	 手 机 内 置 系 统 多 吗 ？ 	 不 多 ， 
第 1300000 行 229142459	安尔雅（anerya）简易衣柜 成人收纳柜非布艺钢架塑料组装单双人储物衣柜衣橱简易衣柜卧室家具家具	真的很好，漂亮时尚	6	门能不能关严会不会被缝隙	会有一点缝的
 ---> 229142459 	 安 尔 雅 （ anerya ） 简 易 衣 柜 成 人 收 纳 柜 非 布 艺 钢 架 塑 料 组 装 单 双 人 储 物 衣 柜 衣 橱 简 易 衣 柜 卧 室 家 具 家 具 	 真 的 很 好 ， 漂 亮 时 尚 	 6 	 门 能 不 能 关 严 会 不 会 被 缝 隙 	 会 有 一 点 缝 的 
第 1400000 行 320560693	小米（mi）小米蓝牙项圈耳机 灰色 动圈+动铁 双单元发声耳机/耳麦影音娱乐数码	此用户未填写评价内容	52	项圈反过来戴能不能正常用	可以
 ---> 320560693 	 小 米 （ mi ） 小 米 蓝 牙 项 圈 耳 机 灰 色 动 圈 + 动 铁 双 单 元 发 声 耳 机 / 耳 麦 影 音 娱 乐 数 码 	 此 用 户 未 填 写 评 价 内 容 	 52 	 项 圈 反 过 来 戴 能 不 能 正 常 用 	 可 以 
第 1500000 行 477009100185	周生生 黄金(足金)爱情密语羽毛项链 86820n 计价黄金项链黄金珠宝首饰	很喜欢，以后会继续买别的	1	链子细不细	还行，吊坠挺重的
 ---> 477009100185 	 周 生 生 黄 金 ( 足 金 ) 爱 情 密 语 羽 毛 项 链 86820 n 计 价 黄 金 项 链 黄 金 珠 宝 首 饰 	 很 喜 欢 ， 以 后 会 继 续 买 别 的 	 1 	 链 子 细 不 细 	 还 行 ， 吊 坠 挺 重 的 
第 1600000 行 64434179	后whoo 津率享红华凝香平颜系列礼盒6件套315ml （水乳+精华+面霜）紧致 补水 保湿 护肤品 套装女 韩国进口套装/礼盒面部护肤美妆护肤	好	17	想问下用过的小仙女们：25岁，敏感肌，容易长痘容易出油皮肤还干，适合用哪个系列？	反正不适合这款，粉色那款还行，总得来说，这个牌子的产品，不是很适合油性皮肤用
 ---> 64434179 	 后 whoo 津 率 享 红 华 凝 香 平 颜 系 列 礼 盒 6 件 套 315 ml （ 水 乳 + 精 华 + 面 霜 ） 紧 致 补 水 保 湿 护 肤 品 套 装 女 韩 国 进 口 套 装 / 礼 盒 面 部 护 肤 美 妆 护 肤 	 好 	 17 	 想 问 下 用 过 的 小 仙 女 们 ： 25 岁 ， 敏 感 肌 ， 容 易 长 痘 容 易 出 油 皮 肤 还 干 ， 适 合 用 哪 个 系 列 ？ 	 反 正 不 适 合 这 款 ， 粉 色 那 款 还 行 ， 总 得 来 说 ， 这 个 牌 子 的 产 品 ， 不 是 很 适 合 油 性 皮 肤 用 
../data/jd/full.skuqa 已全部分词至 C:/Intel/tokenized.skuqa
词频文件已经写入 ../data/jd/vocab.json ../data/jd/vocab.bin
319.10114765167236 秒完成vocab.py

[Info] 原始词库 = 74491
[Info] 频繁字典大小 = 15394, 最低频数 = 20
[Info] 忽略罕词数 = 46307 爆表词汇数 -25509
[Info] 保存词汇到 D:\code\data\jd\reader.data
0.26628804206848145 秒完成vocab.py
'''
