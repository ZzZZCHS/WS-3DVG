import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import PositionWiseFeedForward
import random


class MatchModule(nn.Module):
    def __init__(self, num_proposals=256, lang_size=256, hidden_size=128, lang_num_size=300, det_channel=128, head=4, num_target=8):
        super().__init__()
        self.num_proposals = num_proposals
        self.num_target = num_target
        self.lang_size = lang_size
        self.hidden_size = hidden_size

        self.match = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, 1, 1)
        )
        self.grounding_cross_attn = MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)  # k, q, v


    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """
 
        objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2)  # batch_size, num_proposals, 1
        features = data_dict["bbox_feature"]  # batch_size, num_proposals, feat_size

        batch_size, num_proposal = features.shape[:2]
        len_num_max = data_dict["lang_feat_list"].shape[1]
        target_ids = data_dict["target_ids"].resize(batch_size*len_num_max, self.num_target)
        # features = torch.gather(features, 1, target_ids.unsqueeze(-1).expand(-1, -1, features.shape[-1]))

        # objectness_masks = objectness_masks.permute(0, 2, 1).contiguous()  # batch_size, 1, num_proposals
        # data_dict["random"] = random.random()

        # copy paste
        # feature0 = features.clone()
        # if data_dict["istrain"][0] == 1 and data_dict["random"] < 0.5:
        #     obj_masks = objectness_masks.bool().squeeze(2)  # batch_size, num_proposals
        #     obj_lens = torch.zeros(batch_size, dtype=torch.int).cuda()
        #     for i in range(batch_size):
        #         obj_mask = torch.where(obj_masks[i, :] == True)[0]
        #         obj_len = obj_mask.shape[0]
        #         obj_lens[i] = obj_len
        #
        #     obj_masks_reshape = obj_masks.reshape(batch_size*num_proposal)
        #     obj_features = features.reshape(batch_size*num_proposal, -1)
        #     obj_mask = torch.where(obj_masks_reshape[:] == True)[0]
        #     total_len = obj_mask.shape[0]
        #     obj_features = obj_features[obj_mask, :].repeat(2,1)  # total_len, hidden_size
        #     j = 0
        #     for i in range(batch_size):
        #         obj_mask = torch.where(obj_masks[i, :] == False)[0]
        #         obj_len = obj_mask.shape[0]
        #         j += obj_lens[i]
        #         if obj_len < total_len - obj_lens[i]:
        #             feature0[i, obj_mask, :] = obj_features[j:j + obj_len, :]
        #         else:
        #             feature0[i, obj_mask[:total_len - obj_lens[i]], :] = obj_features[j:j + total_len - obj_lens[i], :]

        feature1 = features[:, None, :, :].repeat(1, len_num_max, 1, 1).reshape(batch_size*len_num_max, num_proposal, -1)
        # feature1 = torch.gather(feature1, 1, target_ids.unsqueeze(-1).expand(-1, -1, feature1.shape[-1]))
        lang_fea = data_dict["lang_fea"]
        # cross-attention
        feature1 = self.grounding_cross_attn(feature1, lang_fea, lang_fea, data_dict["attention_mask"])

        # print("feature1", feature1.shape)
        # match
        feature1_agg = feature1
        feature1_agg = feature1_agg.permute(0, 2, 1).contiguous()

        confidence1 = self.match(feature1_agg).squeeze(1)  # batch_size*len_num_max, num_target
        data_dict["cluster_ref"] = confidence1
        data_dict["target_scores"] = torch.gather(confidence1, 1, target_ids)

        return data_dict


