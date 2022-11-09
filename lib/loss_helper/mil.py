import torch
import sys
import torch.nn.functional as F


def MILNCELoss(data_dict, mask):
    object_feat = data_dict["bbox_feature"]  # bs, num_proposal, dim
    lang_feat = data_dict["lang_emb"]  # bs * len_num_max, dim
    bs, num_proposal, dim = object_feat.shape
    len_num_max = lang_feat.shape[0] // bs
    lang_feat = lang_feat.masked_fill(mask.reshape(bs * len_num_max, 1) == 0, 0.)
    x = torch.matmul(object_feat, lang_feat.t())  # bs, num_proposal, bs * len_num_max
    x = x.permute(0, 2, 1).reshape(bs, bs, len_num_max * num_proposal)  # bs, bs, len_num_max * num_proposal
    nominator = x * torch.eye(x.shape[0])[:, :, None].cuda()
    nominator = nominator.sum(dim=1)  # bs, num_proposal
    nominator = torch.logsumexp(nominator, dim=1)
    denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
    denominator = torch.logsumexp(denominator, dim=1)
    return torch.mean(denominator - nominator)


def MILMARGINLoss(data_dict, mask):
    object_feat = data_dict["bbox_feature"]  # bs, num_proposal, dim
    lang_feat = data_dict["lang_emb"]  # bs * len_num_max, dim
    bs, num_proposal, dim = object_feat.shape
    len_num_max = lang_feat.shape[0] // bs
    lang_feat = lang_feat.masked_fill(mask.reshape(bs * len_num_max, 1) == 0, 0.)
    object_feat = object_feat.unsqueeze(1).expand(-1, bs*len_num_max, -1, -1)  # bs, bs * num, num_proposal, dim
    lang_feat = lang_feat.unsqueeze(1).unsqueeze(0).expand(bs, -1, num_proposal, -1)
    x = F.cosine_similarity(object_feat, lang_feat, dim=-1)  # bs, bs * len_num_max, num_proposal
    x = x.reshape(bs, bs, len_num_max * num_proposal)  # bs, bs, len_num_max * num_proposal
    pos = x * torch.eye(x.shape[0])[:, :, None].cuda()
    x = x - pos
    pos = pos.sum(dim=1)  # bs, num_proposal
    pos = torch.mean(pos, dim=1)
    margin = 0.2
    neg_1 = x.reshape(x.shape[0], -1).sum(dim=1) / (len_num_max * num_proposal * (bs-1))
    neg_2 = x.permute(1, 0, 2).reshape(x.shape[0], -1).sum(dim=1) / (len_num_max * num_proposal * (bs-1))
    # print(pos.mean(), neg_1.mean(), neg_2.mean())
    # sys.exit()
    loss1 = (margin - pos + neg_1).clamp_min(0.)
    loss2 = (margin - pos + neg_2).clamp_min(0.)
    # print(loss1.mean(), loss2.mean())
    return loss1.mean() + loss2.mean()
