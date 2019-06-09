'''
训练后台程序
'''
import time
import math
from tqdm import tqdm
import torch
from eval import eval_performance
import transformer.Constants as Constants
from transformer.Translator import Translator
from dataset import collate_fn
import random
import traceback


def train(model, training_data, validation_data, optimizer, args):
    ''' 开始训练'''
    log_train_file = args.log + '/train.log'
    log_valid_file = args.log + '/valid.log'

    print('[Info] 训练记录写入 {} and {}'.format(log_train_file, log_valid_file))

    with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
        log_tf.write('epoch, loss, ppl, accuracy\n')
        log_vf.write('epoch, loss, ppl, accuracy\n')

    valid_accus = []
    for epoch_i in range(args.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer, args, smoothing=args.label_smoothing)
        print('  - (训练集)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, time: {time:3.3f}秒'.format(
            ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu, time=(time.time() - start)))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, args.device)
        print('  - (验证集) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, time: {time:3.3f}秒'.format(
            ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu,
            time=(time.time() - start)))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': args,
            'epoch': epoch_i}

        if args.save_model:
            model_name = None
            if args.save_mode == 'all':
                model_name = args.log + '/' + args.save_model + '_accu_{accu:3.3f}.ckpt'.format(accu=100 * valid_accu)
            elif args.save_mode == 'best':
                if valid_accu >= max(valid_accus):
                    model_name = args.log + '/' + args.save_model + '.ckpt'
            if model_name:
                torch.save(checkpoint, model_name)
                print('    - [Info] 模型保存于' + model_name)

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch}, {loss: 8.5f}, {ppl: 8.5f}, {accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu))
                log_vf.write('{epoch}, {loss: 8.5f}, {ppl: 8.5f}, {accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu))


def decode(model, src_seq, src_pos, ctx_seq, ctx_pos, args):
    translator = Translator(max_token_seq_len=args.max_token_seq_len, beam_size=5, n_best=1,
                            device=args.device, bad_mask=None, model=model)
    tgt_seq = []
    all_hyp, all_scores = translator.translate_batch(src_seq, src_pos, ctx_seq, ctx_pos)
    for idx_seqs in all_hyp:  # batch
        idx_seq = idx_seqs[0]  # n_best=1
        end_pos = len(idx_seq)
        for i in range(len(idx_seq)):
            if idx_seq[i] == Constants.EOS:
                end_pos = i
                break
        tgt_seq.append([Constants.BOS] + idx_seq[:end_pos][:args.max_word_seq_len] + [Constants.EOS])
    batch_seq, batch_pos = collate_fn(tgt_seq,max_len=args.max_token_seq_len)
    return batch_seq.to(args.device), batch_pos.to(args.device)


def train_epoch(model, training_data, optimizer, args, smoothing):
    ''' Epoch 后台程序'''
    # training_data DataLoader
    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(training_data, mininterval=2, desc='  - (训练)   ', leave=False):
        # prepare data
        ctx_seq, ctx_pos, src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(args.device), batch)
        gold = tgt_seq[:, 1:]  #真正的目标

        error = False
        # if teacher force
        if random.random() < 0.01:  # 每个批次反向传播，不能发散了
            start = random.randint(0, src_pos.shape[0] - 1)
            try:  # 有可能张量长度不一样，丢弃。
                tmp_tgt_seq, tmp_tgt_pos = \
                    decode(model, src_seq=src_seq[start:start + 3], src_pos=src_pos[start:start + 3],
                           ctx_seq=ctx_seq[start:start + 3], ctx_pos=ctx_pos[start:start + 3], args=args)
                tgt_seq[start:start + 3] = tmp_tgt_seq
                tgt_pos[start:start + 3] = tmp_tgt_pos
                print("  ----->teacher force decoding...")
            except Exception as e:
                error = True
                traceback.print_exc(e)
        if error:
            continue


        # forward
        optimizer.zero_grad()
        pred = model(ctx_seq, ctx_pos, src_seq, src_pos, tgt_seq, tgt_pos)
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        # backward
        loss, n_correct = eval_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device):
    ''' Epoch 验证 '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc='  - (验证) ', leave=False):
            # prepare data
            ctx_seq, ctx_pos, src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(ctx_seq, ctx_pos, src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = eval_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy
