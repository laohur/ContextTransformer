from time import time
import os
import random
import json
import sys

'''
此脚本任务：
从skuqa统计商品热度 作为一列 
从jsonl 只保留一条评论 -->review1.jsonl

try catch 是否更慢？

'''


def review1_json(path, review1_json_path, begin=0, end=sys.maxsize):
    '''
    {
	"skuid": "6314359",
	"reviews": ["外观小巧，方正品？"],
	"attributes": [
		["材质", "其它"],
		["风格", "简约现代"],
		["分类", "指甲刀"],
		["name", "777指甲刀套装 指甲剪钳修容组合4件套DS-301（进口）"],
		["item_desc", "暂无信息"],
		["origin", "韩国"],
		["first_cate", "礼品箱包"],
		["second_cate", "礼品"],
		["third_cate", "美妆礼品"]
	]
}
    '''
    if end < 0:
        end = sys.maxsize
    t0 = time()
    print("review1_json正在读取", os.path.abspath(path))
    # doc = open(path, "r", encoding="utf-8").read().splitlines()

    doc = []
    f = open(path, "r", encoding="utf-8")
    line = f.readline()
    print("第一条原始文本", line)
    item_count = 0
    while (line):
        item_count += 1
        if item_count < begin:
            continue
        if item_count > end:
            break

        item = json.loads(line)
        if isinstance(item["reviews"], list) and len(item["reviews"]) >= 1:
            item["reviews"] = item["reviews"][0]
        if item["reviews"] in [None, "", " "]:
            item["reviews"] = "no_review"

        if item["attributes"] in [None, "", " "]:
            continue

        doc.append(json.dumps(item, ensure_ascii=False))
        line = f.readline()
        if item_count % 100000 == 0:
            print("进展", item_count * 100.0 / (end - begin), "第", item_count, "行", "已经装入", len(doc), doc[-1])
            # break

    f.close()
    # random.shuffle(doc)
    print(time() - t0, "秒读出", item_count, "条")
    t0 = time()
    with open(review1_json_path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc))
    print(time() - t0, "秒写入只留一条评论json，共", len(doc), "条。第一条", doc[1])

    return doc

    '''

    效果 100m->0.2m 710条
    id 	 attributes 	 review
    6314359	其它 简约现代 指甲刀条问答商品详情拼接 777指甲刀套装 指甲剪钳修容组合4件套DS-301（进口） 暂无信息 韩国 礼品箱包 礼品 美妆礼品 	外观小巧，方便携带，使用方便
    8921761	304不锈钢 户外 有手柄 旅行壶 6小时-12小时 1.6L-2L 哈尔斯HAERS 1800ml不锈钢真空保温户外运动广口旅行壶暖瓶HG-1800-3（颜色随机） 暂无信息 中国大陆 厨具 水具酒具 保温壶 	还行还行还行还行！！！
    12954702	中国大陆 101g/mL-200g/mL 保湿 任何肤质 国产 成人 面霜 大宝SOD蜜200ml（乳液 面霜 补水保湿 深层滋养 ） 暂无信息 北京 美妆护肤 面部护肤 乳液/面霜 	挺香，还没用上
    17927530	补水 中国大陆 保湿 其它 任何肤质 男女通用 紧肤水 百雀羚 水嫩倍现盈透精华水100ml(补水保湿，提亮肤色) 暂无信息 上海市 美妆护肤 面部护肤 爽肤水/化妆水 	还可以，比较补水的。

    1213.8750610351562 秒读出 972871 条
    7.2417614459991455 秒写入只留一条评论，共 972872 条
    65g->280m
        review1正在读取 D:/code/data/jd/train-anon.jsonl
                1177.8704144954681 秒读出 972871 条
    4.996123790740967 秒写入只留一条评论，共 972872 条。第一条 6314359	其它 简约现代 指甲刀 777指甲刀套装 指甲剪钳修容组合4件套DS-301（进口） 暂无信息 韩国 礼品箱包 礼品 美妆礼品	外观小巧，方便携带，使用方便

    '''


def review1(path, review1_path, begin=0, end=sys.maxsize):
    '''
    {
	"skuid": "6314359",
	"reviews": ["外观小巧，方正品？"],
	"attributes": [
		["材质", "其它"],
		["风格", "简约现代"],
		["分类", "指甲刀"],
		["name", "777指甲刀套装 指甲剪钳修容组合4件套DS-301（进口）"],
		["item_desc", "暂无信息"],
		["origin", "韩国"],
		["first_cate", "礼品箱包"],
		["second_cate", "礼品"],
		["third_cate", "美妆礼品"]
	]
}
    '''
    if end < 0:
        end = sys.maxsize
    t0 = time()
    print("review1正在读取", os.path.abspath(path))
    # doc = open(path, "r", encoding="utf-8").read().splitlines()

    doc = ["id \t attributes \t review"]
    f = open(path, "r", encoding="utf-8")
    line = f.readline()
    print("第一条", line)
    item_count = 0
    while (line):
        item_count += 1
        if item_count < begin:
            continue
        if item_count > end:
            break

        item = json.loads(line)
        review = item["reviews"]
        if review in [None, "", " "]:
            review = "no_review"
            # print("no_review", line)
        attributes = attrs(item["attributes"])
        if attributes in [None, "", " "]:
            # print("no_attributes", line)
            continue
        # line = json.dumps(item, ensure_ascii=False)
        fields = [item["skuid"].strip(), attributes.strip(), review.strip()]
        if len(fields) != 3:
            print("len(fields)!=3", fields)
            continue

        doc.append("\t".join(fields))
        line = f.readline()
        if item_count % 100000 == 0:
            print("进展", item_count * 100.0 / (end - begin), "第", item_count, "行", "已经装入", len(doc), doc[-1])
            # break

    f.close()
    # random.shuffle(doc)
    print(time() - t0, "秒读出", item_count, "条")
    t0 = time()
    with open(review1_path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc))
    print(time() - t0, "秒写入只留一条评论tsv，共", len(doc), "条。第一条", doc[1])

    '''
    
    效果 100m->0.2m 710条
    id 	 attributes 	 review
    6314359	其它 简约现代 指甲刀条问答商品详情拼接 777指甲刀套装 指甲剪钳修容组合4件套DS-301（进口） 暂无信息 韩国 礼品箱包 礼品 美妆礼品 	外观小巧，方便携带，使用方便
    8921761	304不锈钢 户外 有手柄 旅行壶 6小时-12小时 1.6L-2L 哈尔斯HAERS 1800ml不锈钢真空保温户外运动广口旅行壶暖瓶HG-1800-3（颜色随机） 暂无信息 中国大陆 厨具 水具酒具 保温壶 	还行还行还行还行！！！
    12954702	中国大陆 101g/mL-200g/mL 保湿 任何肤质 国产 成人 面霜 大宝SOD蜜200ml（乳液 面霜 补水保湿 深层滋养 ） 暂无信息 北京 美妆护肤 面部护肤 乳液/面霜 	挺香，还没用上
    17927530	补水 中国大陆 保湿 其它 任何肤质 男女通用 紧肤水 百雀羚 水嫩倍现盈透精华水100ml(补水保湿，提亮肤色) 暂无信息 上海市 美妆护肤 面部护肤 爽肤水/化妆水 	还可以，比较补水的。
    
    1213.8750610351562 秒读出 972871 条
    7.2417614459991455 秒写入只留一条评论，共 972872 条
    65g->280m
        review1正在读取 D:/code/data/jd/train-anon.jsonl
                1177.8704144954681 秒读出 972871 条
    4.996123790740967 秒写入只留一条评论，共 972872 条。第一条 6314359	其它 简约现代 指甲刀 777指甲刀套装 指甲剪钳修容组合4件套DS-301（进口） 暂无信息 韩国 礼品箱包 礼品 美妆礼品	外观小巧，方便携带，使用方便

    '''


def attrs(attrs):
    words = []
    for kv_pair in attrs:
        if kv_pair[0] in ["name", "first_cate", "second_cate", "third_cate"]:
            # line += kv_pair[1] + " "
            words.append(kv_pair[1])
    line = words[0] + words[3] + words[2] + words[1]
    return line


def read_context(context_path):
    t0 = time()
    print("read_context正在读取", os.path.abspath(context_path))
    context_doc = open(context_path, "r", encoding="utf-8").read().splitlines()
    # random.shuffle(doc)
    print(time() - t0, "秒读出", len(context_doc), "条商品详情文件，示例", context_doc[1])
    t0 = time()
    sku_context = {}
    for line in context_doc[1:]:
        sents = line.split("\t")
        sku_context[sents[0]] = sents[1:]  # "skuid \t attributes \t review
        if len(sku_context[sents[0]]) != 2:
            print("len(sku_context[sents[0]]) != 2", line)
    print(time() - t0, "秒装载", len(sku_context), "条商品详情")
    return sku_context


'''
7.543272495269775 秒读出 972876 条商品详情文件，示例 6314359	其它 简约现代 指甲刀 777指甲刀套装 指甲剪钳修容组合4件套DS-301（进口） 暂无信息 韩国 礼品箱包 礼品 美妆礼品	外观小巧，方便携带，使用方便
3.2124099731445312 秒装载 972875 条商品详情

'''


def filter(path, sku_counter, min_count=100):
    t0 = time()
    print("review1正在读取", os.path.abspath(path))

    hot = []
    doc = open(path, "r", encoding="utf-8").read().splitlines()
    for line in doc:
        item = line.split("\t")
        if sku_counter[item[0]] < 100:
            continue
        attributes = json.dumps(item["attributes"])
        fields = [item["skuid"], item["reviews"], attributes]
        hot.append(fields)

    with open("data/hot.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(doc))
    print(time() - t0, "秒写入筛出的", len(doc), "条")


def read_skuqa(qa_path):
    '''
614778493287	请问这个复读机中的文件可以通过文件夹分类整理吗，还是所有的mp3都在一个文件夹中，找文件是要逐个翻？	可以分类
191899547	配上	锐龙	AMD	Ryzen	5	1400	，不买显卡，能用吗？	要配独显才能用哦
'''
    t0 = time()
    print("read_skuqa正在读取", os.path.abspath(qa_path))
    qa_doc = open(qa_path, "r", encoding="utf-8").read().splitlines()
    # random.shuffle(doc)
    print(time() - t0, "秒读出", len(qa_doc), "条商品问答。  正在统计商品热度，示例", qa_doc[1])
    t0 = time()
    sku_counter = {}
    qa_doc2 = []
    bad_lines = 0
    for i in range(len(qa_doc)):
        line = qa_doc[i]
        if type(line) != str:
            print("not str  " + str(line))
        sents = line.split("\t")
        if len(sents) != 3:
            # print("len(sents) <= 2", line)
            bad_lines += 1
            continue
        # sents = sents[:3]
        for i in range(3):
            sents[i] = sents[i].strip()

        if sents[0] not in sku_counter:
            sku_counter[sents[0]] = 1
        else:
            sku_counter[sents[0]] += 1
        qa_doc2.append(sents)
    print(time() - t0, "秒统计", len(qa_doc2), "条商品问答。舍弃", bad_lines, "正在统计商品热度。示例", qa_doc2[1])
    t0 = time()

    sku_counter = dict(sorted(sku_counter.items(), key=lambda kv: kv[1], reverse=True))
    with open("../../data/jd/jd_counter.json", "w", encoding="utf-8") as f:  # 特殊字符有问题，仅供人类阅读
        json.dump(sku_counter, f, ensure_ascii=False)
    print(time() - t0, "秒统计", len(sku_counter), "种商品热度")

    return qa_doc2, sku_counter


'''
7.543272495269775 秒读出 972876 条商品详情文件，示例 6314359	其它 简约现代 指甲刀 777指甲刀套装 指甲剪钳修容组合4件套DS-301（进口） 暂无信息 韩国 礼品箱包 礼品 美妆礼品	外观小巧，方便携带，使用方便
3.2124099731445312 秒装载 972875 条商品详情
'''


def combine_data(qa_path, context_path):
    t0 = time()
    qa_doc, sku_counter = read_skuqa(qa_path)
    sku_context = read_context(context_path)
    bad_lines = 0
    full_doc = ["skuid \t attributes \t review  \t counter \t question \t answer"]
    for i in range(len(qa_doc)):
        sents = qa_doc[i]  # [id q a]
        # id + [attrs,review] + [counter] +[q,a]
        if sents[0] not in sku_context:
            continue
        # fields = [sents[0]] + sku_context.get(sents[0], ["no_attrs", "no_review"]) + [            str(sku_counter[sents[0]])] + sents[1:3]
        fields = [sents[0]] + sku_context[sents[0]] + [str(sku_counter[sents[0]])] + sents[1:3]
        if len(fields) != 6:
            print("len(fields)!=6 ", fields)
            bad_lines += 1
            continue
        full_doc.append("\t".join(fields))
    qa_doc = full_doc
    del full_doc
    print(time() - t0, "秒完成", len(qa_doc), "条问答商品详情拼接,丢弃", bad_lines, "。正常示例", qa_doc[1])

    t0 = time()
    combine_path = "../../data/jd/full.skuqa"
    with open(combine_path, "w", encoding="utf-8") as f:
        f.write("\n".join(qa_doc))
    print(time() - t0, "秒完成", len(qa_doc), "条问答商品详情写入", os.path.abspath(combine_path))

    return qa_doc


'''
33.6889533996582 秒完成 1685750 条问答商品详情拼接,丢弃 0 。正常示例 614778493287	家用 电源适配器 插卡/U盘 锂电池 200-399 高考听力 英语学习 教学 USB 外语等级考试 TF卡 其他 USB TF卡 手机充电器 listeneer倾听者 mp3智能复读机可断句录音免磁带 中国大陆 数码 电子教育 复读机	手感非常好，还没往里面倒信息，感觉很好	16	请问这个复读机中的文件可以通过文件夹分类整理吗，还是所有的mp3都在一个文件夹中，找文件是要逐个翻？	可以分类
13.653993844985962 秒完成 1685750 条问答商品详情写入 D:/code/data/jd/full.skuqa
'''


def main():
    # jsonl_path = "../../data/jd/train-anon.jsonl.1000"
    # review1_path = "../../data/jd/review1.1000.jsonl"

    jsonl_path = "../../data/jd/train-anon.jsonl"
    review1_json_path = "../../data/jd/review1.jsonl"
    # review1_json(jsonl_path, review1_json_path=review1_json_path)

    review1_path = "../../data/jd/review1.sku "
    # review1(review1_json_path, review1_path)

    qa_path = "../../data/jd/all-anon.skuqa"
    combine_data(qa_path=qa_path, context_path=review1_path)


if __name__ == '__main__':
    # one_review()
    main()

'''

14.54345989227295 秒读出 972871 条
3.5772502422332764 秒写入只留一条评论tsv，共 972872 条。第一条 6314359	777指甲刀套装 指甲剪钳修容组合4件套DS-301（进口）美妆礼品礼品礼品箱包	外观小巧，方便携带，使用方便


16.496156454086304 秒完成 1685750 条问答商品详情拼接,丢弃 0 。正常示例 614778493287	listeneer倾听者 mp3智能复读机可断句录音免磁带复读机电子教育数码	手感非常好，还没往里面倒信息，感觉很好	16	请问这个复读机中的文件可以通过文件夹分类整理吗，还是所有的mp3都在一个文件夹中，找文件是要逐个翻？	可以分类
6.295363903045654 秒完成 1685750 条问答商品详情写入 D:\code\data\jd\full.skuqa

'''
