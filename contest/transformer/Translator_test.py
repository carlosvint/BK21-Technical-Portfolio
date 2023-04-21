import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs', 
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map', 
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))


    def _model_decode(self, trg_seq, enc_output, src_mask):
        dec_output = self.model.transformer.decoder(trg_seq, enc_output, src_mask)
        return F.softmax(self.model.transformer.final_proj(dec_output), dim=-1)


    def _get_init_state(self, src_seq):
        beam_size = self.beam_size

        num_signals = src_seq.shape[0]
        enc_output, src_mask = self.model.transformer.encoder(src_seq)
        dec_output = self._model_decode(self.init_seq.repeat(num_signals, 1), enc_output, src_mask)
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)
        scores = torch.log(best_k_probs).view(num_signals, beam_size)
        gen_seq = self.blank_seqs.repeat(num_signals, 1).clone().detach()
        gen_seq[:, 1] = best_k_idx[:,0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores, src_mask


    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        #assert len(scores.size()) == 1
        
        beam_size = self.beam_size
        num_windows = dec_output.shape[0]

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        
        scores = torch.log(best_k2_probs).view(num_windows, beam_size, -1) + scores.view(num_windows, beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(num_windows, -1).topk(beam_size)
 
        # Get the corresponding positions of the best k candidiates.
        #best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        #best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        #gen_seq[:, :step] = gen_seq[best_k_r_idxs[:,0], :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k2_idx[:,0]

        return gen_seq, scores


    def translate_sentence(self, enc_output, gen_seq_ini, scores_ini, src_mask):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        #assert enc_output.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx 
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha 


        with torch.no_grad():
            
            ans_idx = 0   # default
            for step in range(2, max_seq_len):    # decode up to max length
                dec_output = self._model_decode(gen_seq_ini[:, :step], enc_output, src_mask)
                #print(dec_output)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq_ini, dec_output, scores_ini, step)
                    # Check if all path finished
                    # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx   
                    # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                    # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha, rounding_mode='floor').max(0)

                    break
        return gen_seq.view(-1).tolist()