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
        if len(word) > 9:  # "13653923571"空值
            continue
        if word in [None, "", " "]:
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
        question, answer, attr = None, None, None
        try:
            question, answer, attr = get_jdpair(row)
        except Exception:
            print(Exception)
        valid = True
        for item in [question, answer, attr]:
            # if item in [None, "", " "] or len(item) < 1 or len(item) > 80:
            if item in [None, "", " "] or len(item) < 1:
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
    # split_test(dir + "/" + source, mydir)

    names = ["test", "valid", "train"]
    marks = ["src", "tgt", "attr"]
    for name in names:
        result = read(mydir + "/" + name + ".txt", begin=0, end=-1)
        for i in range(3):
            path = mydir + "/" + name + "_" + marks[i] + ".txt"
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
    
read正在读取 D:\code\data\jd\big\test.txt
0.003988981246948242 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
可 以 插 户 外 广 场 舞 K 歌 的 音 响 吗 ---> 您 好 。 是 可 以 连 接 的 。 感 谢 您 对 新 科 的 关 注 与 支 持 ， 祝 您 生 活 愉 快 ！ <--- 新 科 （ Shinco ） S 1700 动 圈 麦 克 风 家 庭 KTV 演 唱 卡 拉 OK 会 议 演 讲 专 用 有 线 话 筒 银 色 麦 克 风 影 音 娱 乐 数 码
平均问题长 27.0 平均回答长 59.0 平均详情长 92.0
../data/jd/big/test.txt总计 1000 行，有效问答有1000
0.1515648365020752 秒处理 1000 条
 已写入 D:\code\data\jd\big\test_src.txt
 已写入 D:\code\data\jd\big\test_tgt.txt
 已写入 D:\code\data\jd\big\test_attr.txt
read正在读取 D:\code\data\jd\big\valid.txt
0.004022121429443359 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
各 位 老 铁 ， 这 个 袋 鼠 精 服 用 有 效 果 吗 ？ 求 实 话 。 ---> 您 好 ， 此 商 品 能 有 效 缓 解 精 力 不 济 、 易 疲 劳 ， 肾 虚 、 阳 痿 早 泄 、 性 功 能 减 退 、 增 强 精 子 活 力 哦 。 产 品 本 身 没 有 任 何 副 作 用 ， 有 没 有 效 果 自 己 吃 了 才 知 道 ， 等 您 来 哦 ， 老 铁 <--- Bio island 澳 洲 袋 鼠 精 胶 囊 抗 疲 劳 男 性 保 健 品 增 强 免 疫 营 养 健 康 医 药 保 健
平均问题长 41.0 平均回答长 147.0 平均详情长 64.0
../data/jd/big/valid.txt总计 1000 行，有效问答有999
0.13760042190551758 秒处理 999 条
 已写入 D:\code\data\jd\big\valid_src.txt
 已写入 D:\code\data\jd\big\valid_tgt.txt
 已写入 D:\code\data\jd\big\valid_attr.txt
read正在读取 D:\code\data\jd\big\train.txt
6.898552179336548 秒读出 1682340 条
进展 0.0 读取第 0 行，选取 0
面 料 舒 服 吗 ？ ---> 挺 好 的 <--- 火 的 战 车 亚 麻 套 装 男 七 分 短 半 袖 T 恤 纯 色 五 分 袖 社 会 小 伙 短 袖 2018 夏 季 中 国 风 T 恤 男 装 服 饰 内 衣
平均问题长 11.0 平均回答长 5.0 平均详情长 84.0
进展 5.944101667914928 读取第 100000 行，选取 99990
请 问 这 个 表 有 没 有 心 率 计 的 ---> 没 有 ， 只 有 hr 和 5 有 <--- 佳 明 （ GARMIN ） Fenix 3 蓝 宝 石 镜 面 中 文 版 手 表 GPS 智 能 手 表 游 泳 户 外 手 表 男 女 跑 步 运 动 腕 表 多 功 能 登 山 表 户 外 仪 表 户 外 装 备 运 动 户 外
平均问题长 26.798881899370944 平均回答长 21.54497904811433 平均详情长 92.35034153073777
进展 11.888203335829855 读取第 200000 行，选取 199972
质 量 怎 么 样 ， 味 道 大 么 ? ---> 还 好 ! 没 刺 鼻 的 味 道 ， 刚 安 装 起 来 有 股 淡 淡 的 原 木 味 <--- 木 月 家 具 拉 门 衣 柜 组 合 衣 柜 木 质 衣 橱 大 衣 柜 卧 室 家 具 整 体 衣 柜 衣 柜 卧 室 家 具 家 具
平均问题长 26.841123551679477 平均回答长 21.538437689088028 平均详情长 92.33025958504398
进展 17.832305003744786 读取第 300000 行，选取 299957
大 家 都 是 自 己 装 ， 还 是 找 人 装 的 呀 ---> 自 己 装 ， 一 分 价 钱 一 份 货 。 做 工 好 ， 你 安 装 就 轻 松 <--- 雅 鞍 大 众 polo 专 用 座 套 朗 逸 途 观 l 途 安 新 速 腾 宝 来 蔚 领 嘉 旅 高 尔 夫 7 奥 迪 a 3 凌 渡 探 歌 捷 达 四 季 通 用 全 包 坐 垫 座 套 汽 车 装 饰 汽 车 用 品
平均问题长 26.836037045186327 平均回答长 21.53946219137346 平均详情长 92.34567172737516
进展 23.77640667165971 读取第 400000 行，选取 399945
到 货 通 知 一 会 儿 不 到 半 天 时 间 又 没 货 了 ， 还 没 来 得 及 下 单 呢 ！ 期 望 周 末 到 货 。 ---> 问 客 服 <--- 妙 洁 胶 棉 吸 水 拖 把 替 换 头 拖 把 头 3 入 装 拖 把 / 扫 把 清 洁 用 具 家 庭 清 洁 / 纸 品
平均问题长 26.79842028673871 平均回答长 21.49663704600121 平均详情长 92.34523660694194
进展 29.72050833957464 读取第 500000 行，选取 499930
口 红 润 吗 ？ ---> 不 干 ， 还 可 以 <--- 瓷 妆 亮 泽 丝 滑 口 红 保 湿 不 易 掉 色 唇 膏 女 （ 下 单 后 不 可 修 改 产 品 颜 色 、 收 货 信 息 ） 口 红 香 水 彩 妆 美 妆 护 肤
平均问题长 26.791787266642796 平均回答长 21.461753722013636 平均详情长 92.34869612006457
进展 35.66461000748957 读取第 600000 行，选取 599915
有 防 潮 垫 吗 ---> 没 有 <--- 探 路 者 （ TOREAD ） 户 外 通 款 三 人 一 层 半 帐 篷 防 雨 透 气 伞 架 式 结 构 自 动 速 开 免 搭 建 帐 篷 TEDC 90663 棕 榈 金 / 藏 青 帐 篷 / 垫 子 户 外 装 备 运 动 户 外
平均问题长 26.775961968008854 平均回答长 21.46511844991632 平均详情长 92.34974396415498
进展 41.60871167540449 读取第 700000 行，选取 699900
袜 子 一 个 月 到 现 在 都 没 收 到 ！ ---> 收 到 了 ， 忘 评 了 <--- 南 极 人 【 十 双 装 】 袜 子 男 棉 质 纯 色 船 袜 舒 适 透 气 四 季 学 院 风 青 年 运 动 袜 子 男 十 双 装 休 闲 棉 袜 内 衣 服 饰 内 衣
平均问题长 26.76089332634187 平均回答长 21.452845473859874 平均详情长 92.35385433082679
进展 47.55281334331942 读取第 800000 行，选取 799889
杭 州 有 人 转 吗 ？ 求 一 套 ---> 我 可 以 转 <--- 瑞 典 进 口 BABYBJORN 宝 宝 三 叶 草 学 习 餐 盘 汤 匙 和 叉 子 ( 粉 色 / 紫 色 2 套 装 ) ( 产 地 瑞 典 ) 儿 童 餐 具 喂 养 用 品 母 婴
平均问题长 26.76728925227219 平均回答长 21.477858205503257 平均详情长 92.34742027028716
进展 53.49691501123435 读取第 900000 行，选取 899881
玩 坦 克 世 界 2 k 屏 可 以 最 高 特 效 吗 ？ ---> 可 以 <--- 蓝 宝 石 ( Sapphire ) RX 580 8 G D 5 超 白 金 极 光 特 别 版 1430 MHz / 8400 MHz 8 GB / 256 bit GDDR 5 DX 12 吃 鸡 显 卡 显 卡 电 脑 配 件 电 脑 、 办 公
平均问题长 26.76807181386004 平均回答长 21.496470648373897 平均详情长 92.34187371233118
进展 59.44101667914928 读取第 1000000 行，选取 999867
身 高 153 体 重 100 斤 穿 多 大 号 ---> 小 号 ， 短 宽 短 宽 的 <--- 天 姿 尚 衬 衫 女 2018 春 季 新 款 韩 版 白 色 刺 绣 木 耳 边 宽 松 韩 范 职 业 长 袖 衬 衣 上 衣 女 装 6990 衬 衫 女 装 服 饰 内 衣
平均问题长 26.770049646553346 平均回答长 21.518141394664095 平均详情长 92.33403309236819
进展 65.3851183470642 读取第 1100000 行，选取 1099846
请 问 ， 新 买 回 来 的 相 机 。 发 现 屏 幕 里 有 灰 尘 。 大 家 有 这 样 的 吗 ？ ---> 有 可 能 二 手 翻 新 <--- 佳 能 （ Canon ） EOS 6 D 单 反 套 机 （ EF 24 - 70 mm f / 4 L IS USM 镜 头 ） 单 反 相 机 摄 影 摄 像 数 码
平均问题长 26.77252290545867 平均回答长 21.516846434094923 平均详情长 92.34211213014174
进展 71.32922001497914 读取第 1200000 行，选取 1199834
可 不 可 以 语 音 控 制 ---> 不 可 以 语 音 控 制 。 <--- 索 尼 （ SONY ） 电 视 KD - 65 X 7500 D 65 英 寸 大 屏 超 高 清 4 K 安 卓 智 能 HDR 液 晶 平 板 电 视 机 平 板 电 视 大 家 电 家 用 电 器
平均问题长 26.779583025999408 平均回答长 21.522599357411643 平均详情长 92.3413702717457
进展 77.27332168289406 读取第 1300000 行，选取 1299819
大 家 买 的 米 粉 里 面 有 内 包 装 吗 ， 我 这 个 是 直 接 纸 盒 子 里 装 的 ！ ---> 直 接 在 纸 盒 里 <--- 爱 思 贝 （ EARTH ' S BEST ） 世 界 有 机 婴 儿 地 球 好 米 粉 辅 食 多 种 谷 物 粉 宝 宝 高 铁 营 养 三 段 米 粉 / 菜 粉 营 养 辅 食 母 婴
平均问题长 26.78498715206721 平均回答长 21.533752365712175 平均详情长 92.3406094690034
进展 83.21742335080899 读取第 1400000 行，选取 1399801
正 品 吗 ， 质 量 怎 么 样 ---> 非 常 好 ， 是 正 品 。 <--- 乔 丹 女 鞋 2018 春 夏 秋 季 新 款 运 动 鞋 女 保 暖 舒 适 网 面 透 气 跑 步 鞋 气 垫 增 高 鞋 女 情 侣 鞋 休 闲 小 白 鞋 跑 步 鞋 运 动 鞋 包 运 动 户 外
平均问题长 26.789738120105557 平均回答长 21.538541165107638 平均详情长 92.34751414842957
进展 89.16152501872392 读取第 1500000 行，选取 1499782
安 装 了 office 吗 ---> 安 装 了 word 和 Excel 等 基 础 Office 软 件 。 但 邮 箱 软 件 不 是 以 前 习 惯 的 outlook ， 不 好 用 <--- 联 想 ( Lenovo ) Ideapad 720 S 13 . 3 英 寸 超 轻 薄 笔 记 本 电 脑 ( I 7 - 8550 U 8 G 256 G Office 2016 IPS 1 . 1 Kg ) 香 槟 金 笔 记 本 电 脑 整 机 电 脑 、 办 公
平均问题长 26.788784110768024 平均回答长 21.529103210264417 平均详情长 92.34986127993183
进展 95.10562668663884 读取第 1600000 行，选取 1599767
安 装 麻 烦 吗 ？ 需 不 需 要 打 孔 ？ ---> 安 装 简 单 ， 我 自 己 搞 定 的 ， 原 来 顶 棚 有 孔 ， 螺 丝 直 接 拧 上 就 行 <--- 欧 普 照 明 （ OPPLE ） LED 吸 顶 灯 具 小 卧 室 阳 台 厨 房 卫 浴 灯 饰 蓝 色 现 代 简 约 16 瓦 适 6 - 12 平 吸 顶 灯 灯 饰 照 明 家 装 建 材
平均问题长 26.79151727000415 平均回答长 21.529297373119103 平均详情长 92.35469955643568
../data/jd/big/train.txt总计 1682340 行，有效问答有1682091
271.57970809936523 秒处理 1682091 条
 已写入 D:\code\data\jd\big\train_src.txt
 已写入 D:\code\data\jd\big\train_tgt.txt
 已写入 D:\code\data\jd\big\train_attr.txt
286.9999632835388 秒执行完main()
    
    '''
