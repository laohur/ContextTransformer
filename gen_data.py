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
    if len(sents) != 6 or int(sents[4]) < 20:  #去除冷门商品
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
big

Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/gen_data.py', wdir='D:/code/ContextTransformer')
read正在读取 D:\code\data\jd\full.skuqa
8.331626415252686 秒读出 1684340 条
9.887489557266235 秒  1684340 条
splits_write正在划分训练集 D:\code\data\jd\big
测试集已写入 1000
验证集已写入 1000
训练集已写入 1682340
训练集、验证集、测试集已写入 D:\code\data\jd\big 目录下
read正在读取 D:\code\data\jd\big\test.txt
0.0029921531677246094 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
充电宝自己坏了，不充电了啊。还换吗 ---> 不要买 <--- 半岛铁盒 K20精英版 21200毫安移动电源高倍率动力电芯大容量LED手电数字屏显安卓/苹果通用型充电宝 白色移动电源手机配件手机
平均问题长 17.0 平均回答长 3.0 平均详情长 66.0
../data/jd/big/test.txt总计 1000 行，有效问答有1000
0.002991914749145508 秒处理 1000 条
 已写入 D:\code\data\jd\big\test_src.txt.untoken
 已写入 D:\code\data\jd\big\test_tgt.txt.untoken
 已写入 D:\code\data\jd\big\test_attr.txt.untoken
read正在读取 D:\code\data\jd\big\valid.txt
0.0039904117584228516 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
可以降血糖吗 ---> 似乎没有效果，，，，，，，，，，。 <--- 汤臣倍健（BY-HEALTH） 铬酵母片90片维生素/矿物质营养成分医药保健
平均问题长 6.0 平均回答长 17.0 平均详情长 38.0
../data/jd/big/valid.txt总计 1000 行，有效问答有1000
0.003026723861694336 秒处理 1000 条
 已写入 D:\code\data\jd\big\valid_src.txt.untoken
 已写入 D:\code\data\jd\big\valid_tgt.txt.untoken
 已写入 D:\code\data\jd\big\valid_attr.txt.untoken
read正在读取 D:\code\data\jd\big\train.txt
6.110767841339111 秒读出 1682340 条
进展 0.0 读取第 0 行，选取 0
这个产品可以帮助鱼消化吗？ ---> 不可以帮助鱼的消化，是控制水质的可以分解鱼的粪便 <--- 硝化细菌水鱼缸水质浓缩硝化菌（非鱼药）消化细菌净水剂消毒水族药剂水族宠物生活
平均问题长 13.0 平均回答长 24.0 平均详情长 38.0
进展 5.944101667914928 读取第 100000 行，选取 100000
电量用完时充电还是随时都可以充，哪种不伤电池 ---> 没事的 <--- AMAZFIT 米动手表青春版 曜石黑(智能手表 运动手表 心率/睡眠/GPS/蓝牙/通知 华米科技出品）智能手表智能设备数码
平均问题长 14.165908340916591 平均回答长 11.425085749142509 平均详情长 53.77683223167768
进展 11.888203335829855 读取第 200000 行，选取 200000
这个能不能充气？ ---> 可以～ <--- 上夫创意个性新奇特验钞打火机防风直冲点烟器送礼送老公男友打火机火机烟具礼品箱包
平均问题长 14.165129174354128 平均回答长 11.44116279418603 平均详情长 53.740346298268506
进展 17.832305003744786 读取第 300000 行，选取 300000
你好，茶车能放多少尺寸的电茶壶？ ---> 可以通过咚咚问问卖方客服 <--- 醉匠 茶具花梨木茶车移动简约现代实木茶台电磁炉自动上水茶盘家用功夫茶具整套茶具茶具厨具
平均问题长 14.139482868390438 平均回答长 11.425255249149169 平均详情长 53.72664091119696
进展 23.77640667165971 读取第 400000 行，选取 400000
上排水安装有什么要求吗，和下排水有什么区别 ---> 都一样，如果你有钱，可以考虑购买贵一些的！ <--- 小天鹅（LittleSwan）8公斤变频滚筒洗衣机 1400转电机喷淋无残留 抗菌门封圈 TG80V20DG5洗衣机大 家 电家用电器
平均问题长 14.13927215181962 平均回答长 11.404198989502527 平均详情长 53.72199569501076
进展 29.72050833957464 读取第 500000 行，选取 500000
什么快递 ---> 问客服 <--- 爱尔康傲滴护理液近视隐形眼镜美瞳护理液硅水凝胶眼镜清洁液护理液隐形眼镜医药保健
平均问题长 14.139503720992558 平均回答长 11.413497173005654 平均详情长 53.728166543666916
进展 35.66461000748957 读取第 600000 行，选取 600000
木有床垫，直接铺在木板床代替普通凉席，你们觉得好吗？ ---> 大约会硌得慌。因为薄。在床垫上可以。 <--- 8H凉席 小米生态链可水洗夏季凉感软席可折叠 防滑双面可用凉席其它凉席床上用品家纺
平均问题长 14.147114754808742 平均回答长 11.425350957748403 平均详情长 53.72743545427424
进展 41.60871167540449 读取第 700000 行，选取 700000
亲们那款遮瑕液遮瑕最好 ---> 美宝莲家有一款不错 <--- 美宝莲（MAYBELLINE）定制遮瑕液10 6.8ml（遮瑕液 保湿 轻薄 遮痘印黑眼圈 修容）遮瑕香水彩妆美妆护肤
平均问题长 14.139591229155387 平均回答长 11.423993680009028 平均详情长 53.71963611480555
进展 47.55281334331942 读取第 800000 行，选取 800000
可不可以插耳机 ---> 不可以&hellip;&hellip; <--- 卡西欧（CASIO）手表 G-SHOCK 主题系列 音乐蓝牙智能防震防水运动手表 超强LED照明石英表 GBA-400-1A9日韩表腕表钟表
平均问题长 14.137469828162715 平均回答长 11.43033071208661 平均详情长 53.717621602973
进展 53.49691501123435 读取第 900000 行，选取 900000
会不会滑动呀 ---> 基本不会，但我觉得没的席子凉快 <--- 600D冰丝席凉席三件套1.8m床单款可水洗机洗席子1.5夏季空调软席床单 冰丝席床上用品家纺
平均问题长 14.14742650285944 平均回答长 11.436086182126465 平均详情长 53.713328096302114
进展 59.44101667914928 读取第 1000000 行，选取 1000000
处理器跟骁龙845比怎么样。微信支付可以指纹吗 ---> 支持 <--- 华为 HUAWEI P20  AI智慧全面屏 6GB +64GB 亮黑色 全网通版 移动联通电信4G手机 双卡双待手机手机通讯手机
平均问题长 14.147625852374148 平均回答长 11.438026561973437 平均详情长 53.716218283781714
进展 65.3851183470642 读取第 1100000 行，选取 1100000
这个是只有上衣还是一套啊 ---> 一套 <--- 伊能幸福睡衣女夏仿真丝可爱性感丝绸短袖短裤开胸家居服纯色套装睡衣/家居服内衣服饰内衣
平均问题长 14.15124804431996 平均回答长 11.443688687555738 平均详情长 53.71928570974026
进展 71.32922001497914 读取第 1200000 行，选取 1200000
你好，我前几天买个手机壳小了，能换吗 ---> 能换的啊！联系客服就行了 <--- 驰界 oppor9s手机壳男女款潮磨砂硬壳全包卡通指环r9splus挂绳超薄个性防摔保护套手机壳/保护套手机配件手机
平均问题长 14.149625708645242 平均回答长 11.444642129464892 平均详情长 53.72123856563453
进展 77.27332168289406 读取第 1300000 行，选取 1300000
这个和活性炭有区别吗？ ---> 是不是更好呢，没用过活性炭。 <--- 吾柚 活性炭除甲醛家用新房装修去甲醛  净化活性竹炭包 汽车除味 光触媒纳米矿晶1500克净化除味生活日用家居日用
平均问题长 14.14481219629831 平均回答长 11.437915816987832 平均详情长 53.71722867905486
进展 83.21742335080899 读取第 1400000 行，选取 1400000
准时吗？多久需要调时间？ ---> 机械表通用毛病，还算准时 <--- 罗西尼(ROSSINI)手表 典美时尚系列镶钻皮带自动机械钟表女表礼盒套装516764G01C国表腕表钟表
平均问题长 14.138587758151601 平均回答长 11.436353974032876 平均详情长 53.716463059669245
进展 89.16152501872392 读取第 1500000 行，选取 1500000
找遍了没找到设置时间的地方，是不是质量有问题 ---> 从设置的系统选项更改。 <--- 华为畅享7 3GB+32GB 香槟金 移动联通电信4G手机 双卡双待手机手机通讯手机
平均问题长 14.138551240965839 平均回答长 11.436321042452638 平均详情长 53.71874618750254
进展 95.10562668663884 读取第 1600000 行，选取 1600000
可以用在尼康J5系列微单上吗 ---> 尼康微单采用的是CX卡口，和尼康单反的F卡口规格不一样。需要用转接环才能使用。 <--- 适马（SIGMA）17-70mm F2.8-4 DC MACRO OS HSM｜Contemporary 半画幅 标准变焦镜头 微距防抖 （尼康单反卡口）镜头摄影摄像数码
平均问题长 14.137833038854351 平均回答长 11.436010352493529 平均详情长 53.716968926894424
../data/jd/big/train.txt总计 1682340 行，有效问答有1682340
4.716387510299683 秒处理 1682340 条
 已写入 D:\code\data\jd\big\train_src.txt.untoken
 已写入 D:\code\data\jd\big\train_tgt.txt.untoken
 已写入 D:\code\data\jd\big\train_attr.txt.untoken
33.52625060081482 秒执行完main()
'''

'''
C:\ProgramData\Anaconda3\python.exe "C:\Program Files\JetBrains\PyCharm Professional Edition with Anaconda plugin 2019.1.2\helpers\pydev\pydevconsole.py" --mode=client --port=50175
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\code\\ContextTransformer', 'D:/code/ContextTransformer'])
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.4.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 7.4.0
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/gen_data.py', wdir='D:/code/ContextTransformer')
read正在读取 D:\code\data\jd\full.skuqa
4.704388856887817 秒读出 1684340 条
6.784824371337891 秒  1684340 条
splits_write正在划分训练集 D:\code\data\jd\middle
测试集已写入 1000
验证集已写入 1000
训练集已写入 1682340
训练集、验证集、测试集已写入 D:\code\data\jd\middle 目录下


C:\ProgramData\Anaconda3\python.exe "C:\Program Files\JetBrains\PyCharm Professional Edition with Anaconda plugin 2019.1.2\helpers\pydev\pydevconsole.py" --mode=client --port=50833
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\code\\ContextTransformer', 'D:/code/ContextTransformer'])
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.4.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 7.4.0
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/gen_data.py', wdir='D:/code/ContextTransformer')
read正在读取 D:\code\data\jd\middle\test.txt
0.003988981246948242 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
../data/jd/middle/test.txt总计 1000 行，有效问答有188
0.002111196517944336 秒处理 188 条
 已写入 D:\code\data\jd\middle\test_src.txt.untoken
 已写入 D:\code\data\jd\middle\test_tgt.txt.untoken
 已写入 D:\code\data\jd\middle\test_attr.txt.untoken
read正在读取 D:\code\data\jd\middle\valid.txt
0.0029582977294921875 秒读出 1000 条
进展 0.0 读取第 0 行，选取 0
麻烦你说一下你们的联系方式谢谢 ---> 什么意思？ <--- 杨红樱淘气包马小跳系列（典藏版 套装10册）儿童文学童书图书
平均问题长 15.0 平均回答长 5.0 平均详情长 30.0
../data/jd/middle/valid.txt总计 1000 行，有效问答有194
0.002992391586303711 秒处理 194 条
 已写入 D:\code\data\jd\middle\valid_src.txt.untoken
 已写入 D:\code\data\jd\middle\valid_tgt.txt.untoken
 已写入 D:\code\data\jd\middle\valid_attr.txt.untoken
read正在读取 D:\code\data\jd\middle\train.txt
6.360016584396362 秒读出 1682340 条
进展 0.0 读取第 0 行，选取 0
进展 5.944101667914928 读取第 100000 行，选取 19332
进展 11.888203335829855 读取第 200000 行，选取 38799
进展 17.832305003744786 读取第 300000 行，选取 58019
进展 23.77640667165971 读取第 400000 行，选取 77513
你的颜色对版吗？ ---> 没开花，不知道呢， <--- 春天来了   盆栽蟹爪兰花苗 多肉植物 蟹爪兰带根发货当年开花办公桌阳台植物 苗木花卉绿植农资绿植
平均问题长 11.461542431044714 平均回答长 11.310331036973967 平均详情长 42.42900637304229
进展 29.72050833957464 读取第 500000 行，选取 97146
进展 35.66461000748957 读取第 600000 行，选取 116517
这个一天吃几顿啊？ ---> 我当晚餐，两包，加点芝麻糊。 <--- 明安旭 魔芋代餐粉 代餐饱腹食品早晚代餐魔芋粉 500g大容量内含40小包冲饮谷物饮料冲调食品饮料
平均问题长 11.457706105494431 平均回答长 11.32813814174634 平均详情长 42.44515868792804
进展 41.60871167540449 读取第 700000 行，选取 135746
进展 47.55281334331942 读取第 800000 行，选取 155207
生产日期？ ---> 看产品批次吧，问问客服 <--- 光明 莫斯利安 常温酸牛奶（原味）200g*24家庭装中华老字号牛奶乳品饮料冲调食品饮料
平均问题长 11.454757486727488 平均回答长 11.32404901809185 平均详情长 42.43703288490284
进展 53.49691501123435 读取第 900000 行，选取 174626
进展 59.44101667914928 读取第 1000000 行，选取 194048
进展 65.3851183470642 读取第 1100000 行，选取 213273
味道浓吗，喷多少次才能闻香识人 ---> 留香时间很长吧 <--- 爱马仕（HERMES） 香水女士男士淡香水持久香氛香水香水彩妆美妆护肤
平均问题长 11.459779438656376 平均回答长 11.320671999399833 平均详情长 42.437746748314375
进展 71.32922001497914 读取第 1200000 行，选取 232789
进展 77.27332168289406 读取第 1300000 行，选取 252229
近视眼取下眼镜可以用吗 ---> 不可以，没有近视度数调解功能，可以直接戴眼镜使用 <--- 小鸟看看 Pico Neo VR一体机 基础版VR眼镜智能设备数码
平均问题长 11.463124925663084 平均回答长 11.321551758315822 平均详情长 42.43556674463783
进展 83.21742335080899 读取第 1400000 行，选取 271814
进展 89.16152501872392 读取第 1500000 行，选取 291311
进展 95.10562668663884 读取第 1600000 行，选取 310690
../data/jd/middle/train.txt总计 1682340 行，有效问答有326702
3.994313955307007 秒处理 326702 条
 已写入 D:\code\data\jd\middle\train_src.txt.untoken
 已写入 D:\code\data\jd\middle\train_tgt.txt.untoken
 已写入 D:\code\data\jd\middle\train_attr.txt.untoken
11.53410792350769 秒执行完main()



    '''
