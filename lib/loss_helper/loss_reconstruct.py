import torch
import sys
import torch.nn.functional as F


def reconstruct_score(logit, idx, mask, weights=None):
    """
        logit: [bs, len_num_max, max_des_len, vocab_size]
        idx: [bs, len_num_max, max_des_len]
        mask: [bs, len_num_max, max_des_len]
    """

    bs, len_num_max, max_des_len = logit.shape[:3]
    logit = logit.reshape(bs * len_num_max, max_des_len, -1)
    idx = idx[:, :, :max_des_len].reshape(bs * len_num_max, max_des_len)
    mask = mask[:, :, :max_des_len].reshape(bs * len_num_max, max_des_len)

    eps = 0.1
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # [bs * len_num_max, max_des_len]
    smooth_loss = -logit.mean(dim=-1)  # [bs * len_num_max, max_des_len]
    nll_loss = (1 - eps) * nll_loss + eps * smooth_loss

    # nll_loss = nll_loss.masked_fill(mask == 0, 0)
    nll_loss = nll_loss.masked_fill(mask == 2, 0)
    nll_loss0 = nll_loss.masked_fill(mask == 1, 0)
    nll_loss0 = nll_loss0.sum(dim=-1) / ((mask == 0).int().sum(dim=-1) + 1e-7)
    nll_loss1 = nll_loss.masked_fill(mask == 0, 0)
    nll_loss1 = nll_loss1.sum(dim=-1) / ((mask == 1).int().sum(dim=-1) + 1e-7)
    # nll_loss = nll_loss.sum(dim=-1) / ((mask != 2).int().sum(dim=-1) + 1e-7)  # [bs * len_num_max]
    nll_eps = 0.1
    loss = nll_eps * nll_loss0 + (1 - nll_eps) * nll_loss1
    # nll_loss = nll_loss.mean()
    return loss


# def reconstruct_score(rec_feat, ori_feat, mask):
#     """
#         rec_feat: [bs, len_num_max, max_des_len, emb_size]
#         ori_feat: [bs, len_num_max, max_des_len, emb_size]
#         mask: [bs, len_num_max, max_des_len]
#     """
#     bs, len_num_max, max_des_len = rec_feat.shape[:3]
#     rec_feat = rec_feat.reshape(bs * len_num_max, max_des_len, -1)
#     ori_feat = ori_feat.reshape(bs * len_num_max, max_des_len, -1)
#     mask = mask[:, :, :max_des_len].reshape(bs * len_num_max, max_des_len)
#     # if torch.isnan(rec_feat).any():
#     #     print("rec_feat", rec_feat[0])
#     # if torch.isnan(ori_feat).any():
#     #     print("ori_feat", ori_feat[0])
#
#     loss_ = 1. - F.cosine_similarity(rec_feat, ori_feat, dim=-1)  # bs*len_num_max, max_des_len
#     # loss_ = ((rec_feat - ori_feat) ** 2).mean(dim=-1)
#
#     rec_loss = loss_.masked_fill(mask == 2, 0)
#     rec_loss0 = rec_loss.masked_fill(mask == 1, 0)
#     rec_loss1 = rec_loss.masked_fill(mask == 0, 0)
#     rec_loss0 = rec_loss0.sum(dim=-1) / ((mask == 0).int().sum(dim=-1) + 1e-7)
#     rec_loss1 = rec_loss1.sum(dim=-1) / ((mask == 1).int().sum(dim=-1) + 1e-7)
#     loss = rec_loss0 + 5 * rec_loss1
#     return loss


def reconstruct_loss(rec_score):
    """
        rec_score: [bs*len_num_max, n_candidate]
    """
    # n_candidate = rec_score.shape[1]
    # rewards = torch.linspace(0, 1, n_candidate).to(rec_score.device)
    # idx = torch.argsort(rec_score, dim=-1, descending=True)
    # _, idx = torch.sort(idx, dim=-1)
    # rewards = rewards[idx]
    # rec_loss = rewards * rec_score
    # topks = torch.topk(rec_score, k=4, dim=-1, largest=False)[0]
    return rec_score.mean()
    # return topks.mean()


def weakly_supervised_loss(rec_score, all_score, data_dict):
    """
        candidate_score: [bs*len_num_max, n_candidate]
        rec_score: [bs*len_num_max, n_candidate]
    """
    all_score = -all_score.log_softmax(dim=-1)
    target_ids = data_dict["target_ids"].resize(*rec_score.shape)
    candidate_score = torch.gather(all_score, -1, target_ids)

    n_candidate = candidate_score.shape[1]
    # if data_dict["epoch"] > 0:
    #     rewards = torch.zeros(n_candidate).to(candidate_score.device)
    #     rewards[n_candidate-1] = 1.
    # else:
    #     rewards = torch.linspace(0, 1, n_candidate).to(candidate_score.device)  # pseudo-label by rec_loss
    rewards = torch.linspace(0.1, 1, n_candidate).to(candidate_score.device)
    # rewards = rewards**2

    idx = torch.argsort(rec_score, dim=-1, descending=True)
    _, idx = torch.sort(idx, dim=-1)
    rewards = rewards[idx]
    # if data_dict["epoch"] == 0:
    #     rewards = torch.linspace(1, 0, n_candidate).to(candidate_score.device)
    grounding_loss = (rewards * candidate_score).mean()
    # grounding_loss = candidate_score.mean()
    # grounding_loss = -(rewards * candidate_score.log_softmax(dim=-1)).mean()

    # target_object_loss = -torch.gather(all_score, -1, target_ids).mean()

    weak_loss = grounding_loss
    data_dict["grounding_loss"] = grounding_loss
    return weak_loss


if __name__ == '__main__':
    pass
    # score = torch.rand(10, 5)
    # rec_loss = torch.rand(10, 5)
    # print(rec_loss)
    # weakly_supervised_loss(score, rec_loss)
