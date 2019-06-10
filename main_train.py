# -*- coding: utf-8 -*-
import os
import argparse
import torch
import torch.utils.data
from transformer.Models import ContextTransformer
from transformer.Optim import ScheduledOptim
# from trainer import train
from dataset import SeqDataset, paired_collate_fn, tri_collate_fn
from trainer import train
import json
from Util import *


def main():
    parser = argparse.ArgumentParser(description='main_train.py')
    dir = "../data/jd/middle"
    # dir = "../data/jd/big"
    parser.add_argument('-data_dir', default=dir)
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-en_layers', type=int, default=1)
    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true', default=True)
    parser.add_argument('-proj_share_weight', action='store_true', default=True)
    parser.add_argument('-label_smoothing', action='store_true', default=True)
    parser.add_argument('-log', default="log")
    parser.add_argument('-save_model', default="model")
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-device', action='store_true',
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()
    if not os.path.exists(args.log):
        os.mkdir(args.log)

    print("加载词汇表")
    reader = torch.load(args.data_dir + "/reader.data")
    args.max_token_seq_len = reader['settings']["max_token_seq_len"]
    args.max_word_seq_len = reader['settings']["max_word_seq_len"]

    print("加载验证集数据")
    valid_src = read_file(path=args.data_dir + "/valid_src.txt")
    valid_tgt = read_file(path=args.data_dir + "/valid_tgt.txt")
    valid_ctx = read_file(path=args.data_dir + "/valid_attr.txt")
    valid_src, valid_ctx, valid_tgt = \
        digitalize(src=valid_src, tgt=valid_tgt, ctx=valid_ctx, max_sent_len=args.max_token_seq_len - 2,
                   word2idx=reader['dict']['src'], index2freq=reader["dict"]["frequency"], topk=0)
    # training_data, validation_data = prepare_dataloaders(reader, data, args)
    validation_data = torch.utils.data.DataLoader(
        SeqDataset(
            src_word2idx=reader['dict']['src'],
            tgt_word2idx=reader['dict']['tgt'],
            ctx_word2idx=reader['dict']['ctx'],
            src_insts=valid_src,
            ctx_insts=valid_ctx,
            tgt_insts=valid_tgt
            ),
        num_workers=4,
        batch_size=args.batch_size,
        collate_fn=tri_collate_fn)

    print("加载训练集数据")
    # begin, end = 0, sys.maxsize
    begin, end = 0, 10000
    train_src = read_file(path=args.data_dir + "/train_src.txt", begin=begin, end=end)
    train_tgt = read_file(path=args.data_dir + "/train_tgt.txt", begin=begin, end=end)
    train_ctx = read_file(path=args.data_dir + "/train_attr.txt", begin=begin, end=end)
    train_src, train_ctx, train_tgt = \
        digitalize(src=train_src, tgt=train_tgt, ctx=train_ctx, max_sent_len=args.max_token_seq_len - 2,
                   word2idx=reader['dict']['src'], index2freq=reader["dict"]["frequency"], topk=3)

    training_data = torch.utils.data.DataLoader(
        SeqDataset(
            src_word2idx=reader['dict']['src'],
            tgt_word2idx=reader['dict']['tgt'],
            ctx_word2idx=reader['dict']['ctx'],
            src_insts=train_src,
            ctx_insts=train_ctx,
            tgt_insts=train_tgt
            ),
        num_workers=4,
        batch_size=args.batch_size,
        collate_fn=tri_collate_fn,
        shuffle=True)

    args.src_vocab_size = training_data.dataset.src_vocab_size
    args.tgt_vocab_size = training_data.dataset.tgt_vocab_size
    args.ctx_vocab_size = training_data.dataset.ctx_vocab_size
    args.idx2word = {idx: word for word, idx in reader['dict']['src'].items()}

    print("---准备模型---")
    if args.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table 不同 但共用word embedding.'

    print(args)

    args.model_path = "log/model.ckpt"
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=args.device)
        model_opt = checkpoint['settings']
        transformer = ContextTransformer(
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
        transformer.load_state_dict(checkpoint['model'])
        transformer = transformer.to(args.device)
        print('[Info] 装入模型，继续训练')
    else:
        transformer = ContextTransformer(
            args.ctx_vocab_size,
            args.src_vocab_size,
            args.tgt_vocab_size,
            args.max_token_seq_len,
            tgt_emb_prj_weight_sharing=args.proj_share_weight,
            emb_src_tgt_weight_sharing=args.embs_share_weight,
            d_k=args.d_k,
            d_v=args.d_v,
            d_model=args.d_model,
            d_word_vec=args.d_word_vec,
            d_inner=args.d_inner_hid,
            en_layers=args.en_layers,
            n_layers=args.n_layers,
            n_head=args.n_head,
            dropout=args.dropout).to(args.device)

    optimizer0 = torch.optim.Adam(
        filter(lambda x: x.requires_grad, transformer.parameters()),
        betas=(0.9, 0.98), eps=1e-03)
    args_optimizer = ScheduledOptim(optimizer0, args.d_model, args.n_warmup_steps)

    train(transformer, training_data, validation_data, args_optimizer, args)


if __name__ == '__main__':
    main()
