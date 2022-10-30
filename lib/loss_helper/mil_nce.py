import torch


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
