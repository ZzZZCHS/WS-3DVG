import torch
import sys
import torch.nn.functional as F
from .info_nce import InfoNCE


def contra_loss(data_dict):
    ori_lang_feat = data_dict["ori_lang_feat"]  # bs, num_len_max, dim
    contra_lang_feat = data_dict["contra_lang_feat"]
    ori_lang_feat = ori_lang_feat.mean(dim=1)
    contra_lang_feat = contra_lang_feat.mean(dim=1)
    return InfoNCE(temperature=0.1)(ori_lang_feat, contra_lang_feat)
