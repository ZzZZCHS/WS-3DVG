import torch
import sys


def reconstruct_score(logit, idx, mask, weights=None):
    """
        logit: [bs, len_num_max, max_des_len, vocab_size]
        idx: [bs, len_num_max, max_des_len]
        mask: [bs, len_num_max, max_des_len]
    """

    bs, len_num_max, max_des_len = logit.shape[:3]
    logit = logit.reshape(bs * len_num_max, max_des_len, -1)
    idx = idx.reshape(bs * len_num_max, max_des_len)
    mask = mask.reshape(bs * len_num_max, max_des_len)

    eps = 0.01
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
    loss = nll_loss0 + 5 * nll_loss1
    # nll_loss = nll_loss.mean()
    return loss


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
    return rec_score.mean()

def weakly_supervised_loss(candidate_score, rec_score, all_score, data_dict):
    """
        candidate_score: [bs*len_num_max, n_candidate]
        rec_score: [bs*len_num_max, n_candidate]
    """
    n_candidate = candidate_score.shape[1]
    if data_dict["epoch"] > 5:
        rewards = torch.zeros(n_candidate).to(candidate_score.device)
        rewards[n_candidate-1] = 1.
    else:
        rewards = torch.linspace(0, 1, n_candidate).to(candidate_score.device)  # pseudo-label by rec_loss

    idx = torch.argsort(rec_score, dim=-1, descending=True)
    _, idx = torch.sort(idx, dim=-1)
    rewards = rewards[idx]
    grounding_loss = -(rewards * candidate_score.log_softmax(dim=-1)).mean()

    all_score = all_score.log_softmax(dim=-1)
    target_ids = data_dict["target_ids"].resize(*rec_score.shape)
    target_object_loss = -torch.gather(all_score, -1, target_ids).mean()

    weak_loss = grounding_loss + target_object_loss
    data_dict["grounding_loss"] = grounding_loss
    return weak_loss


if __name__ == '__main__':
    pass
    # score = torch.rand(10, 5)
    # rec_loss = torch.rand(10, 5)
    # print(rec_loss)
    # weakly_supervised_loss(score, rec_loss)
