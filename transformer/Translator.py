''' This module will handle the text generation with beam search.
废话词 您 可 建 猜 百 度 吗 哪 了 问 么 & ？
若有此词， 降低概率值

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import ContextTransformer
from transformer.Beam import Beam


class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, max_token_seq_len, beam_size, n_best, device, bad_mask, model):
        self.max_token_seq_len = max_token_seq_len
        self.beam_size = beam_size
        self.n_best = n_best
        self.device = device
        self.bad_mask = bad_mask.to(self.device) if bad_mask is not None else None
        self.model = model.to(self.device)

    def translate_batch(self, src_seq, src_pos, ctx_seq, ctx_pos):
        # if not src_seq.shape == src_pos.shape == ctx_seq.shape == ctx_pos.shape:
        # print("shapes", src_seq.shape, src_pos.shape, ctx_seq, ctx_pos)

        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(src_seq, src_enc, ctx_seq, ctx_enc, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)

            active_ctx_seq = collect_active_part(ctx_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_ctx_enc = collect_active_part(ctx_enc, active_inst_idx, n_prev_active_inst, n_bm)

            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_ctx_seq, active_ctx_enc, active_inst_idx_to_position_map

        # inst_dec_beams, len_dec_seq, src_seq, src_enc, ctx_seq, ctx_enc, inst_idx_to_position_map, n_bm)

        def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, src_enc, ctx_seq, ctx_enc, inst_idx_to_position_map,
                             n_bm):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
                return dec_partial_pos

            def predict_word(tgt_seq, tgt_pos, ctx_seq, ctx_output, src_seq, src_output, n_active_inst, n_bm):
                dec_output, *_ = self.model.tgt_decoder(tgt_seq, tgt_pos, ctx_seq, ctx_output, src_seq,
                                                        src_output)  # (batch*n_best)*1*512
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                # word_prob = F.log_softmax(self.model.tgt_word_prj(dec_output), dim=1)  # (batch*beam)*vocab
                dec_output = self.model.tgt_word_prj(dec_output)  # (batch*beam)*vocab
                word_prob = F.log_softmax(dec_output, dim=1)  # (batch*beam)*vocab  log(prob)<0
                if self.bad_mask is not None:
                    bad_mask = self.bad_mask.repeat(word_prob.shape[0], 1)
                    word_prob = word_prob.mul(bad_mask)  # (batch*beam)*vocab
                word_prob = word_prob.view(n_active_inst, n_bm, -1)  # n_active_inst=batch

                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            tgt_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            tgt_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(tgt_seq, tgt_pos, ctx_seq, ctx_enc, src_seq, src_enc, n_active_inst, n_bm)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            # Encode
            # src_seq, src_pos, ctx_seq, ctx_pos
            src_seq, src_pos = src_seq.to(self.device), src_pos.to(self.device)
            ctx_seq, ctx_pos = ctx_seq.to(self.device), ctx_pos.to(self.device)
            # src_enc, *_ = self.model.encoder(src_seq, src_pos)
            ctx_enc, *_ = self.model.ctx_encoder(ctx_seq, ctx_pos)
            src_enc, *_ = self.model.src_encoder(src_seq, src_pos, ctx_seq, ctx_enc)

            # if ctx_seq.shape != src_seq.shape:
            #     print("ctx_seq.shape!=src_seq.shape")
            #     print(ctx_seq.shape, src_seq.shape)

            # Repeat data for beam search
            n_bm = self.beam_size
            n_inst, len_s, d_h = src_enc.size()  # batch src_len embed
            src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)  # 4*22 -> 20*22
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)

            n_inst, len_c, d_h = ctx_enc.size()  # batch ctx_len embed
            ctx_seq = ctx_seq.repeat(1, n_bm).view(n_inst * n_bm, len_c)
            ctx_enc = ctx_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_c, d_h)

            # Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            # Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            # Decode
            for len_dec_seq in range(1, self.max_token_seq_len + 1):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_seq, src_enc, ctx_seq, ctx_enc, inst_idx_to_position_map, n_bm)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, src_enc, ctx_seq, ctx_enc, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, ctx_seq, ctx_enc, inst_idx_to_position_map, active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, self.n_best)

        return batch_hyp, batch_scores
