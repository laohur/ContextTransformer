''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
# from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.Layers import ContextLayer, SourceLayer, TargetLayer
import random


class ContextEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_ctx_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.ctx_word_emb = nn.Embedding(n_ctx_vocab, d_word_vec, padding_idx=Constants.PAD)
        # nn.init.xavier_normal_(self.src_word_emb.weight)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0), freeze=True)

        self.layer_stack = nn.ModuleList([
            ContextLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, ctx_seq, ctx_pos, return_attns=False):

        ctx_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=ctx_seq, seq_q=ctx_seq)  # batch*seq*seq
        non_pad_mask = get_non_pad_mask(ctx_seq)  # batch*seq*1

        # -- Forward
        ctx_output = self.ctx_word_emb(ctx_seq) + self.position_enc(ctx_pos)
        # ctx_output = self.ctx_word_emb(ctx_seq)  # batch*seq*512 为了调试分两步
        # ctx_output += self.position_enc(ctx_pos)  # batch*seq*512

        for ctx_layer in self.layer_stack:
            ctx_output, ctx_slf_attn = ctx_layer(ctx_input=ctx_output, non_pad_mask=non_pad_mask,
                                                 slf_attn_mask=slf_attn_mask)
            if return_attns:
                ctx_slf_attn_list += [ctx_slf_attn]  # 64*seq*seq

        if return_attns:
            return ctx_output, ctx_slf_attn_list

        return ctx_output,


class SourceEncoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        # nn.init.xavier_normal_(self.tgt_word_emb.weight)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0), freeze=True)

        self.layer_stack = nn.ModuleList([
            SourceLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, ctx_seq, ctx_output, encoder=None, return_attns=False):

        src_slf_attn_list, ctx_src_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(src_seq)  # batch*seq*1

        slf_attn_mask_subseq = get_subsequent_mask(src_seq)  # batch*seq*seq
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)  # batch*seq*seq
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)  # batch*seq*seq

        ctx_src_attn_mask = get_attn_key_pad_mask(seq_k=ctx_seq, seq_q=src_seq)  # batch*src_seq*ctx_seq

        # -- Forward
        # src_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)  # batch*seq*512
        # src_output, *_ = encoder(src_seq, src_pos)

        if encoder:
            src_output, *_ = encoder(src_seq, src_pos)
        else:
            src_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)  # batch*seq*512

        for src_layer in self.layer_stack:
            src_output, src_slf_attn, ctx_src_attn = src_layer(
                src_output, ctx_output, non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask, ctx_src_attn_mask=ctx_src_attn_mask)
            # batch*src_seq*512  64*src_seq*src_seq   (batch*8)*src_seq*ctx_seq
            if return_attns:
                src_slf_attn_list += [src_slf_attn]
                ctx_src_attn_list += [ctx_src_attn]

        if return_attns:
            return src_output, src_slf_attn_list, ctx_src_attn_list
        return src_output,


class TargetDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        # nn.init.xavier_normal_(self.tgt_word_emb.weight)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0), freeze=True)

        self.layer_stack = nn.ModuleList([
            TargetLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, ctx_seq, ctx_output, src_seq, src_output, encoder=None, return_attns=False):

        src_slf_attn_list, ctx_tgt_attn_list, src_tgt_attn_list = [], [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)  # batch*tgt_seq*1

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)  # batch*tgt_seq*tgt_seq
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)  # batch*tgt_seq*tgt_seq
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)  # batch*tgt_seq*tgt_seq

        ctx_tgt_attn_mask = get_attn_key_pad_mask(seq_k=ctx_seq, seq_q=tgt_seq)  # batch*tgt_seq*ctx_seq
        src_tgt_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)  # batch*tgt_seq*ctx_seq

        # -- Forward
        # tgt_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)  # batch*tgt_seq*512
        # tgt_output, *_ = encoder(tgt_seq, tgt_seq)
        if encoder:
            tgt_output, *_ = encoder(tgt_seq, tgt_seq)
        else:
            tgt_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)  # batch*tgt_seq*512

        for src_layer in self.layer_stack:
            tgt_output, tgt_slf_attn, ctx_tgt_attn, src_tgt_attn = src_layer(
                tgt_input=tgt_output, ctx_output=ctx_output, src_output=src_output, non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask, ctx_tgt_attn_mask=ctx_tgt_attn_mask, src_tgt_attn_mask=src_tgt_attn_mask)

            if return_attns:
                src_slf_attn_list += [tgt_slf_attn]
                ctx_tgt_attn_list += [ctx_tgt_attn]
                src_tgt_attn_list += [src_tgt_attn]
        if return_attns:
            return tgt_output, tgt_slf_attn, ctx_tgt_attn, src_tgt_attn
        return tgt_output,


class ContextTransformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_ctx_vocab, n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048, en_layers=2,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True, emb_src_tgt_weight_sharing=True):

        super().__init__()

        self.ctx_encoder = ContextEncoder(
            n_ctx_vocab=n_ctx_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=en_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
        self.encoder = self.ctx_encoder
        # self.encoder = None
        self.src_encoder = SourceEncoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.tgt_decoder = TargetDecoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.tgt_decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_ctx_vocab == n_src_vocab == n_tgt_vocab, \
                "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.src_encoder.src_word_emb.weight = self.tgt_decoder.tgt_word_emb.weight
        self.ctx_encoder.ctx_word_emb.weight = self.src_encoder.src_word_emb.weight

        #weight会变吗

    def forward(self, src_seq, src_pos, ctx_seq, ctx_pos, tgt_seq, tgt_pos):
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]
        ctx_output = None
        if random.random() < 0.2:
            ctx_output, *_ = self.ctx_encoder(ctx_seq, ctx_pos)  # batch*ctx_seq*512

        encoder = self.encoder if random.random() < 0.1 else None
        src_output, *_ = self.src_encoder(src_seq, src_pos, ctx_seq, ctx_output, encoder=encoder)  # batch*src_seq*512

        encoder = self.encoder if random.random() < 0.1 else None
        tgt_output, *_ = self.tgt_decoder(tgt_seq, tgt_pos, ctx_seq, ctx_output, src_seq, src_output, encoder=encoder)

        seq_logit = self.tgt_word_prj(tgt_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))


# Define some helper functions

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask
