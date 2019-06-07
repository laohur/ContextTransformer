''' Define the Layers '''
import torch.nn as nn
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


def gate(x, y, hidden_size=512):
    assert x.shape == y.shape
    shape = x.shape
    gate_x = nn.Linear(x, y, hidden_size)
    gate_y = nn.Linear(x, y, hidden_size)
    g = nn.functional.sigmoid(gate_x + gate_y)
    return g * x + (1 - g) * y


class ContextLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(ContextLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, ctx_input, non_pad_mask=None, slf_attn_mask=None):
        ctx_output, ctxenc_slf_attn = self.slf_attn(ctx_input, ctx_input, ctx_input, mask=slf_attn_mask)
        ctx_output *= non_pad_mask  # batch*seq*512
        # old = ctx_output

        ctx_output = self.pos_ffn(ctx_output)
        ctx_output *= non_pad_mask  # batch*seq*512
        # ctx_output = gate(old, ctx_output)

        return ctx_output, ctxenc_slf_attn


class SourceLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(SourceLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ctx_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, src_input, ctx_output, non_pad_mask=None, slf_attn_mask=None, ctx_src_attn_mask=None):
        src_output, src_slf_attn = self.slf_attn(src_input, src_input, src_input, mask=slf_attn_mask)
        src_output *= non_pad_mask

        if ctx_output is not None:  # 不能用 if  == ！=
            src_output, ctx_src_attn = self.ctx_attn(src_output, ctx_output, ctx_output, mask=ctx_src_attn_mask)
            src_output *= non_pad_mask
        else:
            ctx_src_attn = None

        src_output = self.pos_ffn(src_output)
        src_output *= non_pad_mask

        return src_output, src_slf_attn, ctx_src_attn


class TargetLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(TargetLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ctx_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.src_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, tgt_input, ctx_output, src_output, non_pad_mask=None, slf_attn_mask=None, ctx_tgt_attn_mask=None,
                src_tgt_attn_mask=None):
        tgt_output, tgt_slf_attn = self.slf_attn(tgt_input, tgt_input, tgt_input, mask=slf_attn_mask)
        tgt_output *= non_pad_mask  # batch*tgt_seq*512 * batch*tgt_seq*1 --> batch*tgt_seq*512

        if ctx_output is not None:
            tgt_output, ctx_tgt_attn = self.ctx_attn(tgt_output, ctx_output, ctx_output, mask=ctx_tgt_attn_mask)  # 64*
            tgt_output *= non_pad_mask
        else:
            ctx_tgt_attn = None

        tgt_output, src_tgt_attn = self.src_attn(tgt_output, src_output, src_output, mask=src_tgt_attn_mask)
        tgt_output *= non_pad_mask

        tgt_output = self.pos_ffn(tgt_output)
        tgt_output *= non_pad_mask

        return tgt_output, tgt_slf_attn, ctx_tgt_attn, src_tgt_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
