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
            if item in [None, "", " "] or len(item) < 5:  # big1 middle5
                valid = False
                break
        if not valid:
            continue

        if len(question) > 30 or len(answer) > 30 or len(attr) > 80:
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

'''
big配置  热度不限 句长不限
middle配置 热度20 句长80
'''


if __name__ == '__main__':
    t0 = time()
    # mydir = "../data/jd/big"
    mydir = "../data/jd/middle"
    main(mydir)
    print(time() - t0, "秒执行完main()")
'''
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/gen_data.py', wdir='D:/code/ContextTransformer')
read正在读取 D:\code\data\jd\full.skuqa
4.288510799407959 秒读出 1684340 条
5.756581544876099 秒  1684340 条
splits_write正在划分训练集 D:\code\data\jd\big
测试集已写入 1000
验证集已写入 1000
训练集已写入 1682340
训练集、验证集、测试集已写入 D:\code\data\jd\big 目录下
read正在读取 D:\code\data\jd\big\test.txt
0.002991914749145508 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
触屏有时候失灵大家都这样吗 ---> 是啊 <--- 努比亚（nubia）Z17mini 6GB+64GB 香槟金 全网通 移动联通电信4G手机 双卡双待手机手机通讯手机
平均问题长 13.0 平均回答长 2.0 平均详情长 58.0
../data/jd/big/test.txt总计 1000 行，有效问答有1000
0.003988981246948242 秒处理 1000 条
 已写入 D:\code\data\jd\big\test_src.txt.untoken
 已写入 D:\code\data\jd\big\test_tgt.txt.untoken
 已写入 D:\code\data\jd\big\test_attr.txt.untoken
read正在读取 D:\code\data\jd\big\valid.txt
0.002991199493408203 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
厚度怎么样？表的质量怎么样？例如表带什么的 ---> 我用的挺好的。带着大气 <--- 西铁城(CITIZEN)手表 自动机械皮表带商务男表NH8350-08AB日韩表腕表钟表
平均问题长 21.0 平均回答长 11.0 平均详情长 44.0
../data/jd/big/valid.txt总计 1000 行，有效问答有1000
0.0029916763305664062 秒处理 1000 条
 已写入 D:\code\data\jd\big\valid_src.txt.untoken
 已写入 D:\code\data\jd\big\valid_tgt.txt.untoken
 已写入 D:\code\data\jd\big\valid_attr.txt.untoken
read正在读取 D:\code\data\jd\big\train.txt
5.65782356262207 秒读出 1682340 条
进展 0.0 读取第 0 行，选取 0
请问你们的bb有没有出现这种情况：之前用普通奶瓶，喝的快，但容易吐奶。用博士后不吐奶，但喝奶速度慢了很多。而且由于奶嘴缓慢出奶，bb不用力去吸，等奶自己流出来。有这种情况吗？有的话，怎样解决？谢谢！ ---> 我觉得不好用奶瓶经常漏奶 <--- 布朗博士（DrBrown’s）宽口径PPSU奶瓶 防胀气婴儿奶瓶150ml 爱宝选WB5111-CH（鸡年纪念版）奶瓶奶嘴喂养用品母婴
平均问题长 99.0 平均回答长 12.0 平均详情长 67.0
进展 5.944101667914928 读取第 100000 行，选取 100000
质量怎么样？ ---> 不错，就感觉大了点。 <--- 星启 诛仙青云志赵丽颖碧瑶同款手链戒子一体伤心花连指手镯COS古装汉服影视剧首饰品137手链/脚链时尚饰品珠宝首饰
平均问题长 14.133138668613315 平均回答长 11.448115518844812 平均详情长 53.656473435265646
进展 11.888203335829855 读取第 200000 行，选取 200000
16年的途观可以用吗？自己可以安装吗 ---> 动力能力强的话，很简单啊。 <--- 马勒（MAHLE）空调滤清器LA621(大众CC/速腾/迈腾/明锐（14年之前）/途安/途观/高尔夫6/帕萨特/奥迪Q3)空调滤清器维修保养汽车用品
平均问题长 14.137894310528447 平均回答长 11.43841280793596 平均详情长 53.68977155114224
进展 17.832305003744786 读取第 300000 行，选取 300000
这个电脑支持内存扩展吗？最多扩展到多大 ---> 没下限 <--- 华为(HUAWEI) MateBook D(2018版) 15.6英寸轻薄微边框笔记本(i5-8250U 8G 128G+1T MX150 2G独显 office)银笔记本电脑整机电脑、办公
平均问题长 14.138839537201543 平均回答长 11.437791874027086 平均详情长 53.69733100889664
进展 23.77640667165971 读取第 400000 行，选取 400000
女生用蓝色合适吗？有没有买过的小姐姐给点意见 ---> 心水极光色可是好久没有货&hellip; <--- 华为 HUAWEI P20 AI智慧全面屏 6GB +64GB 樱粉金 全网通版 移动联通电信4G手机 双卡双待手机手机通讯手机
平均问题长 14.146217134457164 平均回答长 11.454203864490339 平均详情长 53.70762073094817
进展 29.72050833957464 读取第 500000 行，选取 500000
40是多少毫米 ---> 鞋子好像不说毫米，说码数的。 <--- 双星跑步鞋男飞织气垫鞋春夏季新款网面透气减震慢跑鞋运动鞋 M9055 深蓝 40跑步鞋运动鞋包运动户外
平均问题长 14.148801702396595 平均回答长 11.453347093305814 平均详情长 53.70926458147084
进展 35.66461000748957 读取第 600000 行，选取 600000
玩王者帧率是多少呢，如果我开虎牙直播软件玩呢 ---> 个人认为可以开高真，高真60玩王者效果不算好！ <--- vivo X21 全面屏 双摄拍照游戏手机 6GB+128GB 冰钻黑 移动联通电信全网通4G手机 双卡双待手机手机通讯手机
平均问题长 14.153961410064317 平均回答长 11.445774257042904 平均详情长 53.71653547244088
进展 41.60871167540449 读取第 700000 行，选取 700000
这款六岁小朋友玩儿会太简单吗？ ---> 我们也是六岁，可以自己独立完成 <--- 乐高 玩具 小拼砌师 Juniors 4岁-7岁 安娜和艾莎的冰雪乐园 10736 积木LEGO早教启智益智玩具玩具乐器
平均问题长 14.154992635724806 平均回答长 11.440159371200899 平均详情长 53.719976114319834
进展 47.55281334331942 读取第 800000 行，选取 800000
激光去痦子留下的坑可以祛除吗 ---> 不知道 <--- 美德玛（Mederma）平滑凝露20g（疤痕膏 手术疤痕 新老疤痕 烫伤 痘印 儿童  德国原装进口)润肤身体护理个人护理
平均问题长 14.142616071729911 平均回答长 11.426199467250665 平均详情长 53.728566589291766
进展 53.49691501123435 读取第 900000 行，选取 900000
为什么我喝了之后，每天早早起床拉肚子，我喜欢睡前喝一杯，连拉三天了 ---> 我喝没问题 <--- 新西兰原装进口 安佳（Anchor）成人奶粉乳粉 全脂 900g罐装成人奶粉饮料冲调食品饮料
平均问题长 14.141240954176718 平均回答长 11.434902850107944 平均详情长 53.71976697803669
进展 59.44101667914928 读取第 1000000 行，选取 1000000
包安装吗？ ---> 包安装 <--- 韩泰汽车轮胎途虎品质包安装 迎福然 H430轮胎维修保养汽车用品
平均问题长 14.138509861490139 平均回答长 11.433045566954434 平均详情长 53.71645728354272
进展 65.3851183470642 读取第 1100000 行，选取 1100000
装少许的衣物能行么，亲们 ---> 装夏季一套可以。 <--- Landcase 抽绳双肩背包男简约小包束口袋迷你收纳书包 5530黑色休闲运动包功能箱包礼品箱包
平均问题长 14.139465327758794 平均回答长 11.436350512408625 平均详情长 53.72035479967745
进展 71.32922001497914 读取第 1200000 行，选取 1200000
戴着会过敏吗 ---> 没有 <--- 【品牌官方直营】 施华洛世奇  Latisha Flower 链坠  项链时尚饰品珠宝首饰
平均问题长 14.144019046650794 平均回答长 11.437876301769748 平均详情长 53.716973569188696
invalid literal for int() with base 10: ' counter '
进展 77.27332168289406 读取第 1300000 行，选取 1299999
云南总部发货什么意思，怎么不是贵州发货 ---> 茅台自己的店 <--- 【云商总部发货】贵州茅台酒 (新飞天) 53度 500ml（新老包装随机发货！）白酒白酒酒类
平均问题长 14.140693076923077 平均回答长 11.44176 平均详情长 53.716193076923076
进展 83.21742335080899 读取第 1400000 行，选取 1399999
厂家负责安装吗？还是需要自己安装？ ---> 自己装 <--- 戴森(Dyson) 吸尘器 V10 Fluffy 手持吸尘器家用除螨无线吸尘器/除螨仪生活电器家用电器
平均问题长 14.139346428571429 平均回答长 11.432941428571429 平均详情长 53.71854857142857
进展 89.16152501872392 读取第 1500000 行，选取 1499999
3米沙发用几条 ---> 问店家哦 <--- 南极人 夏季沙发垫套装夏天麻将凉席坐垫防滑罩全包欧式组合飘窗椅子巾沙发垫套/椅垫居家布艺家纺
平均问题长 14.137923333333333 平均回答长 11.432742 平均详情长 53.71973466666667
进展 95.10562668663884 读取第 1600000 行，选取 1599999
请问下，是有香的好还是无香的好呢？ ---> 都好 <--- 洁柔（C&S）手帕纸 黑Face 加厚4层面巾纸6片*18包 古龙水香水味（可湿水 超迷你方包装）手帕纸清洁纸品家庭清洁/纸品
平均问题长 14.138903125 平均回答长 11.434696875 平均详情长 53.71009625
../data/jd/big/train.txt总计 1682340 行，有效问答有1682339
4.682474136352539 秒处理 1682339 条
 已写入 D:\code\data\jd\big\train_src.txt.untoken
 已写入 D:\code\data\jd\big\train_tgt.txt.untoken
 已写入 D:\code\data\jd\big\train_attr.txt.untoken
29.96636176109314 秒执行完main()
'''

'''
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/gen_data.py', wdir='D:/code/ContextTransformer')
read正在读取 D:\code\data\jd\full.skuqa
4.439134836196899 秒读出 1684340 条
6.021892547607422 秒  1684340 条
splits_write正在划分训练集 D:\code\data\jd\middle
测试集已写入 1000
验证集已写入 1000
训练集已写入 1682340
训练集、验证集、测试集已写入 D:\code\data\jd\middle 目录下
read正在读取 D:\code\data\jd\middle\test.txt
0.002992391586303711 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
../data/jd/middle/test.txt总计 1000 行，有效问答有177
0.001994609832763672 秒处理 177 条
 已写入 D:\code\data\jd\middle\test_src.txt.untoken
 已写入 D:\code\data\jd\middle\test_tgt.txt.untoken
 已写入 D:\code\data\jd\middle\test_attr.txt.untoken
read正在读取 D:\code\data\jd\middle\valid.txt
0.0030281543731689453 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
../data/jd/middle/valid.txt总计 1000 行，有效问答有174
0.0019936561584472656 秒处理 174 条
 已写入 D:\code\data\jd\middle\valid_src.txt.untoken
 已写入 D:\code\data\jd\middle\valid_tgt.txt.untoken
 已写入 D:\code\data\jd\middle\valid_attr.txt.untoken
read正在读取 D:\code\data\jd\middle\train.txt
5.761622667312622 秒读出 1682340 条
进展 0.0 读取第 0 行，选取 0
进展 5.944101667914928 读取第 100000 行，选取 17609
invalid literal for int() with base 10: ' counter '
进展 11.888203335829855 读取第 200000 行，选取 34961
捷达能用嘛 ---> 不知道，我的是东风锐琪皮卡，刚刚好 <--- 沿途 车载充气床 带头部护档 汽车用后排充气床垫 车震旅行气垫床 家用轿车睡垫 自驾游装备用品 米色 N25车载床安全自驾汽车用品
平均问题长 13.342085693038156 平均回答长 12.152680052628568 平均详情长 55.80464504318975
进展 17.832305003744786 读取第 300000 行，选取 52356
进展 23.77640667165971 读取第 400000 行，选取 69679
进展 29.72050833957464 读取第 500000 行，选取 87095
进展 35.66461000748957 读取第 600000 行，选取 104625
进展 41.60871167540449 读取第 700000 行，选取 122106
进展 47.55281334331942 读取第 800000 行，选取 139470
买小米6X好还是荣耀10好 ---> 用了快一个月，比较流畅！就是为了幻彩后盖！觉得不错！ <--- 荣耀10 全面屏AI摄影手机 6GB+128GB 游戏手机 幻影蓝 全网通 移动联通电信4G 双卡双待手机手机通讯手机
平均问题长 13.364011156441125 平均回答长 12.161072911214518 平均详情长 55.73848326892329
进展 53.49691501123435 读取第 900000 行，选取 156905
进展 59.44101667914928 读取第 1000000 行，选取 174622
进展 65.3851183470642 读取第 1100000 行，选取 192187
进展 71.32922001497914 读取第 1200000 行，选取 209630
进展 77.27332168289406 读取第 1300000 行，选取 227212
买过的朋友们7C好改是7X好呢？ ---> 一分钱一分货小米好 <--- 荣耀 畅玩7X 4GB+64GB 全网通4G全面屏手机 高配版 魅焰红手机手机通讯手机
平均问题长 13.3515115772425 平均回答长 12.158481248872203 平均详情长 55.723699788304366
进展 83.21742335080899 读取第 1400000 行，选取 244524
进展 89.16152501872392 读取第 1500000 行，选取 261838
进展 95.10562668663884 读取第 1600000 行，选取 279440
../data/jd/middle/train.txt总计 1682340 行，有效问答有293418
3.125638723373413 秒处理 293418 条
 已写入 D:\code\data\jd\middle\train_src.txt.untoken
 已写入 D:\code\data\jd\middle\train_tgt.txt.untoken
 已写入 D:\code\data\jd\middle\train_attr.txt.untoken
22.770312786102295 秒执行完main()
'''