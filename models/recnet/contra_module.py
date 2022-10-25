import numpy as np
import torch
import torch.nn as nn
from models.transformer.transformers import Transformer, TransformerDecoder
from models.transformer.attention import MultiHeadAttention
import sys


class ContraModule(nn.Module):
    def __init__(self, hidden_size=128, max_des_len=100, head=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = 0.1

        # self.trans = TransformerDecoder(num_layers=num_decoder_layers, d_model=hidden_size, num_heads=head, dropout=self.dropout)
        self.cross_attention = MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)

    def forward(self, data_dict):
        lang_feat = data_dict["lang_fea"]  # bs*len_num_max, max_des_len, dim
        object_feat = data_dict["bbox_feature"]  # bs, num_proposal, dim
        max_des_len = lang_feat.shape[1]
        bs, num_proposal, dim = object_feat.shape
        len_num_max = lang_feat.shape[0] // bs
        lang_feat = lang_feat.reshape(bs, len_num_max, max_des_len, dim)
        mask = data_dict["all_masks_list"][:, :, :max_des_len].unsqueeze(-1)
        lang_feat = lang_feat.masked_fill(mask == 2, -float("inf"))
        lang_feat = lang_feat.max(dim=2)[0]
        data_dict["ori_lang_feat"] = lang_feat
        # objectness_preds_batch = torch.argmax(data_dict['objectness_scores'], 2).long()
        # attention_mask = (objectness_preds_batch == 0).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, num_proposal
        lang_feature, att_weight = self.cross_attention(lang_feat, object_feat, object_feat, output_attn=True)
        data_dict["contra_lang_feat"] = lang_feature  # bs, num_len_max, dim
        data_dict["contra_att_weight"] = att_weight.mean(dim=1)  # bs, num_len_max, num_proposal
        # print(lang_feature[0][0])
        # if torch.isnan(lang_feat).any():
        #     print(lang_feature)
        # print(torch.min(lang_feature))
        return data_dict
