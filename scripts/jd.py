"获取京东数据，分字以便于通用各种语言"
#614778493287	请问这个复读机中的文件可以通过文件夹分类整理吗，还是所有的mp3都在一个文件夹中，找文件是要逐个翻？	可以分类
# 191899547	配上	锐龙	AMD	Ryzen	5	1400	，不买显卡，能用吗？	要配独显才能用哦
# 199410817	请问你们的蓝牙能充满电能听多少个小时啊~~我开60音量不间断大概能听6个小时左右还有一半的电！想问问大家的是怎样？国外购买，国外保修一年255美刀	2小时


import argparse
import torch
import os

def split_chinese(line):
    # print("之前",line)
    # if "%" in line:
    #     print(line)
    words=[]
    for word in line:
        if ord(word) >=0x4E00 and ord(word)<=0x9fff:
            word=" "+word+" "
        words.append(word)
    line2="".join(words)
    line2=" ".join(line2.split())
    # print("之后",line2)
    return line2

def pure(line):
    line=line.strip('\n')
    line=line.strip('\r')
    return line.strip()

def read(path):
    with open(path, mode="r", encoding="utf8") as f:
        line_count = 0  #总行数
        qlenth = 0 #问题长度
        alenth = 0 # 回答长度
        questions=[]
        answers=[]
        for row in f.readlines():
            line_count += 1
            words = row.split("\t")
            #数据多，出错丢弃.如果数据出错，前面几段都归为问题，最后一个归为回答.
            if len(words)!=3:
                continue
            question=pure(words[1])
            answer= pure(words[2])

            qlenth += len(question)
            alenth += len(answer)
            questions.append(split_chinese(question))
            answers.append(split_chinese(answer))
            # if line_count%100==0:
            #     print(words[1],words[2],line_count)

        assert len(answers)==len(questions)
        print(str(path)+"总计"+str(line_count)+"行，有效问答有"+str(len(answers)))
        print("平均问题长", qlenth / len(answers),"平均回答长", alenth / len(answers))
        # print(word2index)
        return questions,answers

def split(data,train_rate=0.8):
    train_len=int(len(data)*train_rate)
    valid_len=int( (len(data)-train_len )/2 )
    return data[:train_len], data[train_len:train_len+valid_len],data[train_len+valid_len:]

def write(data,path):
    print(data[0:5])
    with open(path, 'w', encoding="utf8") as f:
        for line in data:
            line=pure(line)
            f.write(line+"\n")
        print("被写入"+os.path.abspath(path))
        f.close()    

def main(required=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-source', default=required["source"])
    parser.add_argument('-dir', default=required["dir"])
    args = parser.parse_args()

    questions,answers=read(args.source)
    train_src, valid_src, test_src=split(questions,0.9)
    train_tgt, valid_tgt, test_tgt=split(answers,0.9)

    write(train_src,args.dir+"/"+"train_src.txt")
    write(train_tgt,args.dir+"/"+"train_tgt.txt")
    write(valid_src,args.dir+"/"+"valid_src.txt")
    write(valid_tgt,args.dir+"/"+"valid_tgt.txt")
    write(test_src,args.dir+"/"+"test_src.txt")
    write(test_tgt,args.dir+"/"+"test_tgt.txt")

if __name__ == '__main__':
    required={
        "source": "jd/all-anon.skuqa",
        "dir": "data"
    }
    main(required)



