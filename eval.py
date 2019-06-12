'''
评估后台程序
'''
import torch
import torch.nn.functional as F
import transformer.Constants as Constants
import random


def eval_performance(pred, gold, smoothing=False, args=None):
    ''' 标签平滑'''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1] #2688*7196 ->
    gold = gold.contiguous().view(-1) #128*21 ->2688
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    # start = random.randint(0, gold.shape[0] - 4)
    if args != None and random.random() < 0.001:
        # for i in range(start, start + 3):
        gold = ''.join([args.idx2word[idx.item()] for idx in gold[:20]])
        pred = ''.join([args.idx2word[idx.item()] for idx in pred[:20]])
        print("  [eval_performance]---", loss.item(), pred, '--应为-->', gold)

    return loss, n_correct


def cal_loss(pred, gold, smoothing=False):
    ''' 交叉熵损失 '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD)

    return loss
