# ContextTransformer
A PyTorch implementation of ContextTransformer for Question and Answer.
![avatar](ContextTransformer.png)

## Dependencies
- Python 3.x
- PyTorch >= 0.4 
- tqdm
- numpy

## Dataset
jd:https://github.com/gsh199449/productqa
tb:https://github.com/cooelf/DeepUtteranceAggregation

## Quick Start
* scripts/jdc.py
组装数据源

* get_data.py
 划分训练集 划分问答背景

* vocab.py 
```
文件分词 生成词汇表
```

* retrive 
检索答案


* main_train.py
```
模型训练
```

* main_test.py
```
模型测试
```

## References
https://github.com/siat-nlp/transformer-pytorch

## 更新记录


下一步

下两步
样本级别权重
多样性：我们发现若输出为（包含）高频序列，以及输出序列长度过短或过长，此输出序列的损失项优化都容易导致通用回复，因此我们设计了有效的方法对这些损失项乘以一个较小的权重。有兴趣的读者可以阅读我们的论文及具体的权重计算方式。
https://yq.aliyun.com/articles/174784
Chat More: Deepening and Widening the Chatting Topic via A
https://github.com/jiweil/Jiwei-Thesis

## 词显著度算法
TF-IDF->局部词频/sqrt(全局词频)
