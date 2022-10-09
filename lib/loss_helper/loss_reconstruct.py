import torch


def reconstruct_loss(logit, idx, mask, weights=None):
    """
        logit: [bs, len_num_max, max_des_len, vocab_size]
        idx: [bs, len_num_max, max_des_len]
        mask: [bs, len_num_max, max_des_len]
    """

    bs, len_num_max, max_des_len = logit.shape[:3]
    logit = logit.reshape(bs * len_num_max, max_des_len, -1)
    idx = idx.reshape(bs * len_num_max, max_des_len)
    mask = mask.reshape(bs * len_num_max, max_des_len)

    eps = 0.1
    logit = logit.log_softmax(dim=-1)
    # print(logit[0][0])
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # [bs * len_num_max, max_des_len]
    smooth_loss = -logit.sum(dim=-1)  # [bs * len_num_max, max_des_len]
    # print(nll_loss[13], smooth_loss[13])
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss

    nll_loss = nll_loss.masked_fill(mask == 0, 0)
    nll_loss = nll_loss.masked_fill(mask == 2, 0)
    nll_loss = nll_loss.sum(dim=-1) / ((mask == 1).int().sum(dim=-1) + 1e-7)  # [bs * len_num_max]
    # if torch.isnan(nll_loss).any():
        # print(logit[0][0][:100])
        # print(nll_loss, smooth_loss)
        # print(mask[13])
        # print(nll_loss)
    # nll_loss = nll_loss.mean()
    return nll_loss.contiguous()


def weakly_supervised_loss(score, rec_loss):
    """
        score: [bs*len_num_max, n_candidate]
        rec_loss: [bs*len_num_max, n_candidate]
    """
    n_candidate = score.shape[1]
    rewards = torch.linspace(0, 1, n_candidate).to(score.device)  # pseudo-label by rec_loss

    idx = torch.argsort(rec_loss, dim=-1, descending=True)
    _, idx = torch.sort(idx, dim=-1)
    rewards = rewards[idx]
    grounding_loss = -(rewards * score.log_softmax(dim=-1))
    return grounding_loss.mean()


if __name__ == '__main__':
    score = torch.rand(10, 5)
    rec_loss = torch.rand(10, 5)
    print(rec_loss)
    weakly_supervised_loss(score, rec_loss)
