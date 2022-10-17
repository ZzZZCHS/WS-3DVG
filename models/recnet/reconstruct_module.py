import numpy as np
import torch
import torch.nn as nn
from models.transformer.transformers import Transformer, TransformerDecoder


class ReconstructModule(nn.Module):
    def __init__(self, vocab_size, emb_size=300, hidden_size=128, max_des_len=100, head=4,
                 num_encoder_layers=2, num_decoder_layers=3):
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.mask_vec = torch.nn.Parameter(torch.empty(hidden_size), requires_grad=True)
        nn.init.xavier_normal_(self.mask_vec.unsqueeze(0))

        # self.token_fc = nn.Linear(emb_size, hidden_size)
        # self.word_fc = nn.Linear(emb_size, hidden_size)
        # self.object_fc = nn.Linear(hidden_size, emb_size)

        self.word_position = SinusoidalPositionalEmbedding(hidden_size, 0, max_des_len)

        # self.reconstruct_trans = Transformer(hidden_size, num_heads=head, num_encoder_layers=num_encoder_layers,
        #                                      num_decoder_layers=num_decoder_layers, dropout=0.3)
        self.reconstruct_trans = TransformerDecoder(num_layers=num_decoder_layers, d_model=hidden_size, num_heads=head, dropout=0.2)

        self.vocab_fc = nn.Linear(hidden_size, vocab_size)
        # self.rec_fc = nn.Linear(hidden_size, emb_size)

    def forward(self, original_embs, object_feat, masks_list):
        """
            original_embs: [bs, len_num_max, max_des_len, emb_dim]
            object_feat: [bs, len_num_max, 1+proposal, dim]
            data_dict['XXX_masks_list']: [bs, len_num_max, max_des_len]
        """
        max_des_len = original_embs.shape[2]
        batch_size, len_num_max, object_num = object_feat.shape[:3]

        object_feat = object_feat.reshape(batch_size * len_num_max, object_num, -1)
        word_embs = original_embs.reshape(batch_size * len_num_max, max_des_len, -1)
        # word_embs = self.word_fc(word_embs)
        all_masks_list = masks_list.reshape(batch_size * len_num_max, max_des_len)
        all_masked_embs = self._mask_words(word_embs, all_masks_list) + self.word_position(word_embs)

        # object_feat = self.object_fc(object_feat)

        embs_mask = 1 - (all_masks_list == 2).int()
        object_mask = torch.zeros(batch_size * len_num_max, object_num).to(object_feat.device)
        # print(all_masked_embs.shape, embs_mask.shape, object_feat.shape, object_mask.shape)
        trans_out = self.reconstruct_trans(object_feat, object_mask, all_masked_embs, embs_mask)
        # print(trans_out.shape, self.hidden_size)
        word_logit = self.vocab_fc(trans_out).reshape(batch_size, len_num_max, max_des_len, self.vocab_size)  #[bs, len_num_max, max_des_len, vocab_size]
        # rec_word_feat = self.rec_fc(trans_out).reshape(batch_size, len_num_max, max_des_len, self.emb_size)
        # rec_word_feat = trans_out.reshape(batch_size, len_num_max, max_des_len, self.hidden_size)
        return word_logit

    def _mask_words(self, words_feat, masks_list):
        """
            words_feat: [bs, n, hidden_dim]
            mask_list: [bs, n]  #0:word  1:masked  2:padding
        """
        token = self.mask_vec.cuda().unsqueeze(0)
        # token = self.token_fc(token)  # [bs, n, hidden_dim]

        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill(masks_list.unsqueeze(-1) != 1, 0)
        words_feat1 = words_feat.masked_fill(masks_list.unsqueeze(-1) == 1, 0) + masked_words_vec
        # words_feat1 = masked_words_vec
        return words_feat1


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
