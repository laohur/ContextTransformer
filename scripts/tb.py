import argparse
import torch
import os


def split_chinese(line):
    # print("之前",line)
    # if "%" in line:
    #     print(line)
    words = []
    for word in line:
        if ord(word) >= 0x4E00 and ord(word) <= 0x9fff:
            word = " " + word + " "
        words.append(word)
    line2 = "".join(words)
    line2 = " ".join(line2.split())
    # print("之后",line2)
    return line2


def pure(line):
    line = line.strip('\n')
    line = line.strip('\r')
    return line.strip()


def read(path):
    with open(path, mode="r", encoding="utf8") as f:
        line_count = 0  # 总行数
        qlenth = 0  # 问题长度
        alenth = 0  # 回答长度
        questions = []
        answers = []
        for row in f.readlines():
            line_count += 1
            words = row.split("\t")
            # 数据多，出错丢弃.如果数据出错，前面几段都归为问题，最后一个归为回答.
            if len(words) < 3 or int(words[0]) != 1:
                continue
            question = pure(words[1])
            answer = pure(words[2])

            qlenth += len(question)
            alenth += len(answer)
            questions.append((question))
            answers.append((answer))
            # if line_count%100==0:
            #     print(words[1],words[2],line_count)

        assert len(answers) == len(questions)
        print(str(path) + "总计" + str(line_count) + "行，有效问答有" + str(len(answers)))
        print("平均问题长", qlenth / len(answers), "平均回答长", alenth / len(answers))
        # print(word2index)
        return questions, answers


def split(data, train_rate=0.8):
    train_len = int(len(data) * train_rate)
    valid_len = int((len(data) - train_len) / 2)
    return data[:train_len], data[train_len:train_len + valid_len], data[train_len + valid_len:]


def write(data, path):
    print(data[0:5])
    with open(path, 'w', encoding="utf8") as f:
        for line in data:
            line = pure(line)
            f.write(line + "\n")
        print("被写入" + os.path.abspath(path))
        f.close()

def splits_write(x, suffix, dir, shuffle=True):
    print("splits_write正在划分训练集", os.path.abspath(dir))
    # if shuffle:
    #     random.shuffle(x)
    test_len, valid_len = 100, 1000
    right = len(x) - test_len
    left = right - valid_len

    with open(dir + "/test" + suffix, "w", encoding="utf-8") as f:
        f.write("\n".join(x[:test_len]))
    print("测试集已写入")
    with open(dir + "/valid" + suffix, "w", encoding="utf-8") as f:
        f.write("\n".join(x[test_len:valid_len]))
    print("验证集已写入")
    with open(dir + "/train" + suffix, "w", encoding="utf-8") as f:
        f.write("\n".join(x[valid_len:]))
    print("训练集、验证集、测试集已写入", dir, "目录下")

def main():
    dir = "../../data/tb"
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', default=dir)
    parser.add_argument('-source', default=dir + "/train.txt")
    args = parser.parse_args()

    questions, answers = read(args.source)
    splits_write(questions, dir=dir, suffix="_src.txt")
    splits_write(answers, dir=dir, suffix="_tgt.txt")

    # train_src, valid_src, test_src = split(questions, 0.9)
    # train_tgt, valid_tgt, test_tgt = split(answers, 0.9)
    #
    # write(train_src, args.dir + "/" + "train_src.txt")
    # write(train_tgt, args.dir + "/" + "train_tgt.txt")
    # write(valid_src, args.dir + "/" + "valid_src.txt")
    # write(valid_tgt, args.dir + "/" + "valid_tgt.txt")
    # write(test_src, args.dir + "/" + "test_src.txt")
    # write(test_tgt, args.dir + "/" + "test_tgt.txt")


if __name__ == '__main__':
    main()
