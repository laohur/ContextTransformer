# -*- coding: utf-8 -*-
import os
import torch
import sys
import torch.utils.data
import argparse
from tqdm import tqdm
from dataset import SeqDataset, paired_collate_fn, tri_collate_fn
from transformer.Translator import Translator
import json
from Util import *
from transformer.Models import ContextTransformer
from retrive import cdscore, bleu;
from retrive import cdscore


def main():
    # test_path="../data/tb/test_src.txt"
    # test_path = "../data/qa_data/test_src.txt"
    data_dir = "../data/jd/pure"
    parser = argparse.ArgumentParser(description='main_test.py')
    parser.add_argument('-model_path', default="log/model.ckpt", help='模型路径')
    parser.add_argument('-data_dir', default=data_dir, help='模型路径')
    parser.add_argument('-src', default=data_dir + "/test_src.txt", help='测试集源文件路径')
    parser.add_argument('-data', default=data_dir + "/reader.data", help='训练数据')
    parser.add_argument('-output_dir', default="output", help="输出路径")
    parser.add_argument('-beam_size', type=int, default=10, help='Beam size')
    parser.add_argument('-batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-n_best', type=int, default=3, help="""多句输出""")
    parser.add_argument('-device', action='store_true',
                        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print("加载词汇表", os.path.abspath(args.data))
    reader = torch.load(args.data)
    args.max_token_seq_len = reader['settings']["max_token_seq_len"]

    test_src = read_file(path=args.data_dir + "/test_src.txt")
    test_ctx = read_file(path=args.data_dir + "/test_attr.txt")
    test_tgt = read_file(path=args.data_dir + "/test_tgt.txt")
    test_src, test_ctx, _ = digitalize(src=test_src, tgt=None, ctx=test_ctx, max_sent_len=20,
                                       word2idx=reader['dict']['src'], index2freq=reader["dict"]["frequency"], topk=0)

    test_loader = torch.utils.data.DataLoader(
        SeqDataset(
            src_word2idx=reader['dict']['src'],
            tgt_word2idx=reader['dict']['tgt'],
            ctx_word2idx=reader['dict']['ctx'],
            src_insts=test_src,
            ctx_insts=test_ctx),
        num_workers=4,
        batch_size=args.batch_size,
        collate_fn=paired_collate_fn)

    bad_words = ['您', '建', '猜', '查', '吗', '哪', '了', '问', '么', '&', '？']
    bad_idx = [0, 1, 2, 3] + [reader['dict']['src'][w] for w in bad_words]
    # 最后一个批次不等长
    bads = torch.ones((1, len(reader['dict']['tgt'])))
    for i in bad_idx:
        bads[0][i] = 100  # log(prob)<0  分别观察 0.01  100
    args.bad_mask = bads
    # args.bad_mask = None

    checkpoint = torch.load(args.model_path)
    model_opt = checkpoint['settings']
    args.model = ContextTransformer(
        model_opt.ctx_vocab_size,
        model_opt.src_vocab_size,
        model_opt.tgt_vocab_size,
        model_opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_tgt_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        en_layers=model_opt.en_layers,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout)
    args.model.load_state_dict(checkpoint['model'])
    args.model.word_prob_prj = torch.nn.LogSoftmax(dim=1)
    print('[Info] Trained model state loaded.')

    translator = Translator(max_token_seq_len=args.max_token_seq_len, beam_size=args.beam_size, n_best=args.n_best,
                            device=args.device, bad_mask=args.bad_mask, model=args.model)
    # def __init__(self, max_token_seq_len, beam_size, n_best, device, bad_mask, model):

    # translator.model_opt = checkpoint['settings']
    translator.model.eval()

    path = args.output_dir + '/test_out.txt'
    predicts = [];

    for batch in tqdm(test_loader, mininterval=0.1, desc='  - (Test)', leave=False):
        all_hyp, all_scores = translator.translate_batch(*batch)
        for idx_seqs in all_hyp:  # batch
            answers = []
            for idx_seq in idx_seqs:  # n_best
                end_pos = len(idx_seq)
                for i in range(len(idx_seq)):
                    if idx_seq[i] == Constants.EOS:
                        end_pos = i
                        break
                pred_line = ' '.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq[:end_pos]])
                answers.append(pred_line)
            # f.write("\t".join(answers) + '\n')
            # answers_line = "\t".join(answers)
            predicts.append(answers)

    # with open(path, 'w', encoding="utf-8") as f:
    #     f.write("\n".join(predicts))
    # print('[Info] 测试完成，文件写入' + path)

    docBleu = 0.0
    docCdscore = 0.0
    for i in range(len(test_tgt)):
        # for answer in predicts[i].split("\t"):
        print(test_tgt[i] + "----->" + "_".join(predicts[i]))
        # bleu = get_moses_multi_bleu([test_tgt[i]] * 3, predicts[i], lowercase=True)
        bleu_score = bleu([test_tgt[i]], predicts[i]);
        docBleu += bleu_score
        cdscore = cdscore([test_tgt[i]], predicts[i])
        docCdscore += cdscore
        print(" cd_score:",cdscore," bleu:",bleu_score)
    docBleu /= len(test_tgt)
    docCdscore /= len(test_tgt)
    print(" doc bleu-->" + str(docBleu) + "   docCdscore-->" + str(docCdscore))


if __name__ == "__main__":
    main()
    sys.exit()
