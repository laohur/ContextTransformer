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
    count_file(mydir)
    main(mydir)
    print(time() - t0, "秒完成vocab.py")
'''
C:\ProgramData\Anaconda3\python.exe "C:\Program Files\JetBrains\PyCharm Professional Edition with Anaconda plugin 2019.1.2\helpers\pydev\pydevconsole.py" --mode=client --port=54032
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\code\\ContextTransformer', 'D:/code/ContextTransformer'])
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.4.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 7.4.0
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/vocab.py', wdir='D:/code/ContextTransformer')
token_file分词 D:\code\data\jd\big\test_src.txt.untoken 将写入 D:\code\data\jd\big\test_src.txt
第一条 触屏有时候失灵大家都这样吗
../data/jd/big/test_src.txt.untoken 已全部分词至 ../data/jd/big/test_src.txt
token_file分词 D:\code\data\jd\big\test_tgt.txt.untoken 将写入 D:\code\data\jd\big\test_tgt.txt
第一条 是啊
../data/jd/big/test_tgt.txt.untoken 已全部分词至 ../data/jd/big/test_tgt.txt
token_file分词 D:\code\data\jd\big\test_attr.txt.untoken 将写入 D:\code\data\jd\big\test_attr.txt
../data/jd/big/test_attr.txt.untoken 不计入词频
第一条 努比亚（nubia）Z17mini 6GB+64GB 香槟金 全网通 移动联通电信4G手机 双卡双待手机手机通讯手机
../data/jd/big/test_attr.txt.untoken 已全部分词至 ../data/jd/big/test_attr.txt
token_file分词 D:\code\data\jd\big\valid_src.txt.untoken 将写入 D:\code\data\jd\big\valid_src.txt
第一条 厚度怎么样？表的质量怎么样？例如表带什么的
../data/jd/big/valid_src.txt.untoken 已全部分词至 ../data/jd/big/valid_src.txt
token_file分词 D:\code\data\jd\big\valid_tgt.txt.untoken 将写入 D:\code\data\jd\big\valid_tgt.txt
第一条 我用的挺好的。带着大气
../data/jd/big/valid_tgt.txt.untoken 已全部分词至 ../data/jd/big/valid_tgt.txt
token_file分词 D:\code\data\jd\big\valid_attr.txt.untoken 将写入 D:\code\data\jd\big\valid_attr.txt
../data/jd/big/valid_attr.txt.untoken 不计入词频
第一条 西铁城(CITIZEN)手表 自动机械皮表带商务男表NH8350-08AB日韩表腕表钟表
../data/jd/big/valid_attr.txt.untoken 已全部分词至 ../data/jd/big/valid_attr.txt
token_file分词 D:\code\data\jd\big\train_src.txt.untoken 将写入 D:\code\data\jd\big\train_src.txt
第一条 请问你们的bb有没--->有出现这种情况：之前用普通奶瓶，喝的快，但容易吐奶。用博士后不吐奶，但喝奶速度慢了很多。而且由于奶嘴缓慢出奶，bb不用力去吸，等奶自己流出来。有这种情况吗？有的话，怎样解决？谢谢！
第 100000 行 索尼55x9000e可以挂吗
 ---> 索 尼 55 x 9000 e 可 以 挂 吗 
已统计词频 6523
第 200000 行 为什么有的防伪码是贝泰妮，有的是薇诺娜，你们买的也是这样吗？
 ---> 为 什 么 有 的 防 伪 码 是 贝 泰 妮 ， 有 的 是 薇 诺 娜 ， 你 们 买 的 也 是 这 样 吗 ？ 
已统计词频 8274
第 300000 行 请问这个香水有手袋吗？
 ---> 请 问 这 个 香 水 有 手 袋 吗 ？ 
已统计词频 9463
第 400000 行 是不是每晚上都贴是吗，还是隔几天贴一次，大姨妈可以贴吗
 ---> 是 不 是 每 晚 上 都 贴 是 吗 ， 还 是 隔 几 天 贴 一 次 ， 大 姨 妈 可 以 贴 吗 
已统计词频 10483
第 500000 行 是手机使用吗？
 ---> 是 手 机 使 用 吗 ？ 
已统计词频 11331
第 600000 行 煮粥煲汤会溢出来吗
 ---> 煮 粥 煲 汤 会 溢 出 来 吗 
已统计词频 12122
第 700000 行 这个吸管可以配吗？
 ---> 这 个 吸 管 可 以 配 吗 ？ 
已统计词频 12859
第 800000 行 装驱动的时候提示说设备未连接，怎么回事？设备管理器能看到这个硬盘信息，但是到了我的电脑里还是没有该盘。
 ---> 装 驱 动 的 时 候 提 示 说 设 备 未 连 接 ， 怎 么 回 事 ？ 设 备 管 理 器 能 看 到 这 个 硬 盘 信 息 ， 但 是 到 了 我 的 电 脑 里 还 是 没 有 该 盘 。 
已统计词频 13520
第 900000 行 为什么一百多条好评都是买的杏色m号？&hellip;&hellip;无语=_=&hellip;
 ---> 为 什 么 一 百 多 条 好 评 都 是 买 的 杏 色 m 号 ？ & hellip ; & hellip ; 无 语 = _ = & hellip ; 
已统计词频 14072
第 1000000 行 17平方的房子装多大的空调？
 ---> 17 平 方 的 房 子 装 多 大 的 空 调 ？ 
已统计词频 14627
第 1100000 行 怎么好像都没有前置摄像头啊？还是我没有看到？
 ---> 怎 么 好 像 都 没 有 前 置 摄 像 头 啊 ？ 还 是 我 没 有 看 到 ？ 
已统计词频 15158
第 1200000 行 是紫砂壶吗？
 ---> 是 紫 砂 壶 吗 ？ 
已统计词频 15690
第 1300000 行 云南总部发货什么意思，怎么不是贵州发货
 ---> 云 南 总 部 发 货 什 么 意 思 ， 怎 么 不 是 贵 州 发 货 
已统计词频 16177
第 1400000 行 厂家负责安装吗？还是需要自己安装？
 ---> 厂 家 负 责 安 装 吗 ？ 还 是 需 要 自 己 安 装 ？ 
已统计词频 16617
第 1500000 行 3米沙发用几条
 ---> 3 米 沙 发 用 几 条 
已统计词频 17092
第 1600000 行 请问下，是有香的好还是无香的好呢？
 ---> 请 问 下 ， 是 有 香 的 好 还 是 无 香 的 好 呢 ？ 
已统计词频 17534
../data/jd/big/train_src.txt.untoken 已全部分词至 ../data/jd/big/train_src.txt
token_file分词 D:\code\data\jd\big\train_tgt.txt.untoken 将写入 D:\code\data\jd\big\train_tgt.txt
第一条 我觉得不好用奶瓶经常漏奶
第 100000 行 索尼65x7500挂的挺好，而且墙面还是空心砖
 ---> 索 尼 65 x 7500 挂 的 挺 好 ， 而 且 墙 面 还 是 空 心 砖 
已统计词频 18405
第 200000 行 不知道啊，没有注意
 ---> 不 知 道 啊 ， 没 有 注 意 
已统计词频 18868
第 300000 行 有
 ---> 有 
已统计词频 19308
第 400000 行 我两天贴一次！感觉挺好的！至少喝酒也没胖肚子！我一般都是贴上去后！用热宝放在独自上热一会儿！之后拿走！那个贴就像再燃烧脂肪一样！挺舒服的
 ---> 我 两 天 贴 一 次 ！ 感 觉 挺 好 的 ！ 至 少 喝 酒 也 没 胖 肚 子 ！ 我 一 般 都 是 贴 上 去 后 ！ 用 热 宝 放 在 独 自 上 热 一 会 儿 ！ 之 后 拿 走 ！ 那 个 贴 就 像 再 燃 烧 脂 肪 一 样 ！ 挺 舒 服 的 
已统计词频 19685
第 500000 行 是滴
 ---> 是 滴 
已统计词频 20083
第 600000 行 不会
 ---> 不 会 
已统计词频 20482
第 700000 行 不知道有没有得配。
 ---> 不 知 道 有 没 有 得 配 。 
已统计词频 20807
第 800000 行 这个盘不装系统买它干嘛？这么强大的读写性能就是为系统准备的！！！装系统的时候会提示初始化和格式化，然后装系统，我用光盘装的系统才10分钟左右，装好系统再到三星官网下驱动
 ---> 这 个 盘 不 装 系 统 买 它 干 嘛 ？ 这 么 强 大 的 读 写 性 能 就 是 为 系 统 准 备 的 ！ ！ ！ 装 系 统 的 时 候 会 提 示 初 始 化 和 格 式 化 ， 然 后 装 系 统 ， 我 用 光 盘 装 的 系 统 才 10 分 钟 左 右 ， 装 好 系 统 再 到 三 星 官 网 下 驱 动 
已统计词频 21144
第 900000 行 也许胖的人少
 ---> 也 许 胖 的 人 少 
已统计词频 21507
第 1000000 行 我装了一匹的
 ---> 我 装 了 一 匹 的 
已统计词频 21830
第 1100000 行 右下角
 ---> 右 下 角 
已统计词频 22165
第 1200000 行 还没用呢、感觉是
 ---> 还 没 用 呢 、 感 觉 是 
已统计词频 22467
第 1300000 行 茅台自己的店
 ---> 茅 台 自 己 的 店 
已统计词频 22790
第 1400000 行 自己装
 ---> 自 己 装 
已统计词频 23071
第 1500000 行 问店家哦
 ---> 问 店 家 哦 
已统计词频 23339
第 1600000 行 都好
 ---> 都 好 
已统计词频 23606
../data/jd/big/train_tgt.txt.untoken 已全部分词至 ../data/jd/big/train_tgt.txt
token_file分词 D:\code\data\jd\big\train_attr.txt.untoken 将写入 D:\code\data\jd\big\train_attr.txt
../data/jd/big/train_attr.txt.untoken 不计入词频
第一条 布朗博士（DrBrown’s）宽口径PPSU奶瓶 防胀气婴儿奶瓶150ml 爱宝选WB5111-CH（鸡年纪念版）奶瓶奶嘴喂养用品母婴
第 100000 行 nb 757-l400（32-70英寸）电视挂架 电视架 电视机挂架 电视支架 旋转伸缩 乐视海信海尔tcl康佳三星夏普lg家电配件大 家 电家用电器
 ---> nb 757 - l 400 （ 32 - 70 英 寸 ） 电 视 挂 架 电 视 架 电 视 机 挂 架 电 视 支 架 旋 转 伸 缩 乐 视 海 信 海 尔 tcl 康 佳 三 星 夏 普 lg 家 电 配 件 大 家 电 家 用 电 器 
第 200000 行 薇诺娜（winona）多效舒敏润养特护套装（特护霜50g+面膜20ml×2+防晒乳15g+柔肤水30ml+洁面15g+特护霜2g×5）套装/礼盒面部护肤美妆护肤
 ---> 薇 诺 娜 （ winona ） 多 效 舒 敏 润 养 特 护 套 装 （ 特 护 霜 50 g + 面 膜 20 ml × 2 + 防 晒 乳 15 g + 柔 肤 水 30 ml + 洁 面 15 g + 特 护 霜 2 g × 5 ） 套 装 / 礼 盒 面 部 护 肤 美 妆 护 肤 
第 300000 行 卡尔文克雷恩（calvin klein）绝色魅影女士淡香水30ml（又名卡尔文克雷恩绝色魅影女士香水）香水香水彩妆美妆护肤
 ---> 卡 尔 文 克 雷 恩 （ calvin klein ） 绝 色 魅 影 女 士 淡 香 水 30 ml （ 又 名 卡 尔 文 克 雷 恩 绝 色 魅 影 女 士 香 水 ） 香 水 香 水 彩 妆 美 妆 护 肤 
第 400000 行 【买5送5！10盒100贴】懒人贴倩滋体膜肚脐贴 水桶腰大肚腩啤酒肚/10贴/盒润肤身体护理个人护理
 ---> 【 买 5 送 5 ！ 10 盒 100 贴 】 懒 人 贴 倩 滋 体 膜 肚 脐 贴 水 桶 腰 大 肚 腩 啤 酒 肚 / 10 贴 / 盒 润 肤 身 体 护 理 个 人 护 理 
第 500000 行 三星（samsung）存储卡32gb 读速95mb/s  class10 高速tf卡（micro sd卡）红色plus升级版存储卡数码配件数码
 ---> 三 星 （ samsung ） 存 储 卡 32 gb 读 速 95 mb / s class 10 高 速 tf 卡 （ micro sd 卡 ） 红 色 plus 升 级 版 存 储 卡 数 码 配 件 数 码 
第 600000 行 美的（midea）一锅双胆 七段调压 收汁入味yl50simple102 5l高压锅电压力锅厨房小电家用电器
 ---> 美 的 （ midea ） 一 锅 双 胆 七 段 调 压 收 汁 入 味 yl 50 simple 102 5 l 高 压 锅 电 压 力 锅 厨 房 小 电 家 用 电 器 
第 700000 行 迪士尼（disney）儿童保温杯带吸管不锈钢水杯子男女学生双盖保温水壶送杯套 550ml hc6002a 漫威红色保温杯水具酒具厨具
 ---> 迪 士 尼 （ disney ） 儿 童 保 温 杯 带 吸 管 不 锈 钢 水 杯 子 男 女 学 生 双 盖 保 温 水 壶 送 杯 套 550 ml hc 6002 a 漫 威 红 色 保 温 杯 水 具 酒 具 厨 具 
第 800000 行 三星(samsung) 960 evo 250g m.2 nvme 固态硬盘ssd固态硬盘电脑配件电脑、办公
 ---> 三 星 ( samsung ) 960 evo 250 g m . 2 nvme 固 态 硬 盘 ssd 固 态 硬 盘 电 脑 配 件 电 脑 、 办 公 
第 900000 行 歌米拉 冰丝短袖t恤女2018春季新款韩版条纹薄款短款长袖t恤上衣女t恤女装服饰内衣
 ---> 歌 米 拉 冰 丝 短 袖 t 恤 女 2018 春 季 新 款 韩 版 条 纹 薄 款 短 款 长 袖 t 恤 上 衣 女 t 恤 女 装 服 饰 内 衣 
第 1000000 行 奥克斯（aux）空调大1匹/正1.5匹 变频冷暖智能壁挂式家用空调挂机空调大 家 电家用电器
 ---> 奥 克 斯 （ aux ） 空 调 大 1 匹 / 正 1 . 5 匹 变 频 冷 暖 智 能 壁 挂 式 家 用 空 调 挂 机 空 调 大 家 电 家 用 电 器 
第 1100000 行 小米mix2s 全面屏游戏手机 6gb+64gb 黑色 全网通4g 陶瓷手机手机手机通讯手机
 ---> 小 米 mix 2 s 全 面 屏 游 戏 手 机 6 gb + 64 gb 黑 色 全 网 通 4 g 陶 瓷 手 机 手 机 手 机 通 讯 手 机 
第 1200000 行 藏壶天下 紫砂壶茶壶全手工名家泡茶壶宜兴紫砂功夫茶具石瓢壶茶壶茶具厨具
 ---> 藏 壶 天 下 紫 砂 壶 茶 壶 全 手 工 名 家 泡 茶 壶 宜 兴 紫 砂 功 夫 茶 具 石 瓢 壶 茶 壶 茶 具 厨 具 
第 1300000 行 【云商总部发货】贵州茅台酒 (新飞天) 53度 500ml（新老包装随机发货！）白酒白酒酒类
 ---> 【 云 商 总 部 发 货 】 贵 州 茅 台 酒 ( 新 飞 天 ) 53 度 500 ml （ 新 老 包 装 随 机 发 货 ！ ） 白 酒 白 酒 酒 类 
第 1400000 行 戴森(dyson) 吸尘器 v10 fluffy 手持吸尘器家用除螨无线吸尘器/除螨仪生活电器家用电器
 ---> 戴 森 ( dyson ) 吸 尘 器 v 10 fluffy 手 持 吸 尘 器 家 用 除 螨 无 线 吸 尘 器 / 除 螨 仪 生 活 电 器 家 用 电 器 
第 1500000 行 南极人 夏季沙发垫套装夏天麻将凉席坐垫防滑罩全包欧式组合飘窗椅子巾沙发垫套/椅垫居家布艺家纺
 ---> 南 极 人 夏 季 沙 发 垫 套 装 夏 天 麻 将 凉 席 坐 垫 防 滑 罩 全 包 欧 式 组 合 飘 窗 椅 子 巾 沙 发 垫 套 / 椅 垫 居 家 布 艺 家 纺 
第 1600000 行 洁柔（c&s）手帕纸 黑face 加厚4层面巾纸6片*18包 古龙水香水味（可湿水 超迷你方包装）手帕纸清洁纸品家庭清洁/纸品
 ---> 洁 柔 （ c & s ） 手 帕 纸 黑 face 加 厚 4 层 面 巾 纸 6 片 * 18 包 古 龙 水 香 水 味 （ 可 湿 水 超 迷 你 方 包 装 ） 手 帕 纸 清 洁 纸 品 家 庭 清 洁 / 纸 品 
../data/jd/big/train_attr.txt.untoken 已全部分词至 ../data/jd/big/train_attr.txt
词频文件已经写入 ../data/jd/big/counter.json ../data/jd/big/counter.bin
[Info] 原始词库 = 23835
[Info] 频繁字典大小 = 8908, 最低频数 = 5
[Info] 忽略罕词数 = 12320 爆表词汇数 -76165
[Info] 保存词汇到 D:\code\data\jd\big\reader.json
[Info] 保存词汇到 D:\code\data\jd\big\reader.data
203.54521894454956 秒完成vocab.py

[Info] 原始词库 = 23835
[Info] 频繁字典大小 = 7196, 最低频数 = 9
[Info] 忽略罕词数 = 14032 爆表词汇数 -76165
[Info] 保存词汇到 D:\code\data\jd\big\reader.json
[Info] 保存词汇到 D:\code\data\jd\big\reader.data
0.09541440010070801 秒完成vocab.py
'''

'''
C:\ProgramData\Anaconda3\python.exe "C:\Program Files\JetBrains\PyCharm Professional Edition with Anaconda plugin 2019.1.2\helpers\pydev\pydevconsole.py" --mode=client --port=54258
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\code\\ContextTransformer', 'D:/code/ContextTransformer'])
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.4.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 7.4.0
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/code/ContextTransformer/vocab.py', wdir='D:/code/ContextTransformer')
token_file分词 D:\code\data\jd\middle\test_src.txt.untoken 将写入 D:\code\data\jd\middle\test_src.txt
第一条 怎么没人人那
../data/jd/middle/test_src.txt.untoken 已全部分词至 ../data/jd/middle/test_src.txt
token_file分词 D:\code\data\jd\middle\test_tgt.txt.untoken 将写入 D:\code\data\jd\middle\test_tgt.txt
第一条 你要问啥？
../data/jd/middle/test_tgt.txt.untoken 已全部分词至 ../data/jd/middle/test_tgt.txt
token_file分词 D:\code\data\jd\middle\test_attr.txt.untoken 将写入 D:\code\data\jd\middle\test_attr.txt
../data/jd/middle/test_attr.txt.untoken 不计入词频
第一条 九阳（Joyoung） 免滤全钢多功能家用豆浆机DJ13B-C660SG豆浆机厨房小电家用电器
../data/jd/middle/test_attr.txt.untoken 已全部分词至 ../data/jd/middle/test_attr.txt
token_file分词 D:\code\data\jd\middle\valid_src.txt.untoken 将写入 D:\code\data\jd\middle\valid_src.txt
第一条 说真的，只有一点点辣，还比不上国民女神老干妈，远不如生大蒜
../data/jd/middle/valid_src.txt.untoken 已全部分词至 ../data/jd/middle/valid_src.txt
token_file分词 D:\code\data\jd\middle\valid_tgt.txt.untoken 将写入 D:\code\data\jd\middle\valid_tgt.txt
第一条 众口难调，我吃感觉也还不错啦！
../data/jd/middle/valid_tgt.txt.untoken 已全部分词至 ../data/jd/middle/valid_tgt.txt
token_file分词 D:\code\data\jd\middle\valid_attr.txt.untoken 将写入 D:\code\data\jd\middle\valid_attr.txt
../data/jd/middle/valid_attr.txt.untoken 不计入词频
第一条 韩国三养（SAMYANG）方便面 火鸡面 超辣鸡肉味拌面 700g（140g*5包入）方便食品进口食品食品饮料
../data/jd/middle/valid_attr.txt.untoken 已全部分词至 ../data/jd/middle/valid_attr.txt
token_file分词 D:\code\data\jd\middle\train_src.txt.untoken 将写入 D:\code\data\jd\middle\train_src.txt
第一条 定时预约最长时间是多久？
第 100000 行 我身高180体重80穿多大的比较好呢？
 ---> 我 身 高 180 体 重 80 穿 多 大 的 比 较 好 呢 ？ 
已统计词频 6000
第 200000 行 这店不是山灵的，山灵的店又不卖东西，真是日了
 ---> 这 店 不 是 山 灵 的 ， 山 灵 的 店 又 不 卖 东 西 ， 真 是 日 了 
已统计词频 7524
../data/jd/middle/train_src.txt.untoken 已全部分词至 ../data/jd/middle/train_src.txt
token_file分词 D:\code\data\jd\middle\train_tgt.txt.untoken 将写入 D:\code\data\jd\middle\train_tgt.txt
第一条 要看盖不盖
第 100000 行 我176，xl合适
 ---> 我 176 ， xl 合 适 
已统计词频 9565
第 200000 行 亲，我们是山灵数码旗舰店，感谢您的关注！
 ---> 亲 ， 我 们 是 山 灵 数 码 旗 舰 店 ， 感 谢 您 的 关 注 ！ 
已统计词频 10434
../data/jd/middle/train_tgt.txt.untoken 已全部分词至 ../data/jd/middle/train_tgt.txt
token_file分词 D:\code\data\jd\middle\train_attr.txt.untoken 将写入 D:\code\data\jd\middle\train_attr.txt
../data/jd/middle/train_attr.txt.untoken 不计入词频
第一条 长虹（CHANGHONG）电磁炉整板触控大功率电池炉电磁灶8档火力电磁炉厨房小电家用电器
第 100000 行 探路者（toread）防晒衣 男女士透气户外风衣 tief透湿防紫外线皮肤衣 防晒服 taeg81739户外风衣户外鞋服运动户外
 ---> 探 路 者 （ toread ） 防 晒 衣 男 女 士 透 气 户 外 风 衣 tief 透 湿 防 紫 外 线 皮 肤 衣 防 晒 服 taeg 81739 户 外 风 衣 户 外 鞋 服 运 动 户 外 
第 200000 行 山灵m3s便携无损音乐播放器支持平衡输出hifi蓝牙发烧mp3mp3/mp4影音娱乐数码
 ---> 山 灵 m 3 s 便 携 无 损 音 乐 播 放 器 支 持 平 衡 输 出 hifi 蓝 牙 发 烧 mp 3 mp 3 / mp 4 影 音 娱 乐 数 码 
../data/jd/middle/train_attr.txt.untoken 已全部分词至 ../data/jd/middle/train_attr.txt
词频文件已经写入 ../data/jd/middle/counter.json ../data/jd/middle/counter.bin
[Info] 原始词库 = 11068
[Info] 频繁字典大小 = 4115, 最低频数 = 9
[Info] 忽略罕词数 = 6268 爆表词汇数 -88932
[Info] 保存词汇到 D:\code\data\jd\middle\reader.json
[Info] 保存词汇到 D:\code\data\jd\middle\reader.data
35.688405990600586 秒完成vocab.py
替换
'''