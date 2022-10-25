"""
Modified from: https://github.com/facebookresearch/votenet/blob/master/models/proposal_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

import lib.pointnet2.pointnet2_utils
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes
# from utils.box_util import get_3d_box_batch_of_rois_tensor, rotz_batch_pytorch
from utils.box_util import get_3d_box_torch, rotz_batch_pytorch
from models.proposal_module.ROI_heads.roi_heads import StandardROIHeads


class ProposalModule(nn.Module):
    def __init__(self, num_proposal, sampling, seed_feat_dim=256, dataset_config=None, hidden_size=128, num_target=32):
        super().__init__()
        if hasattr(dataset_config, 'sun_num_class'):
            self.num_class = dataset_config.sun_num_class
            self.num_heading_bin = dataset_config.sun_num_heading_bin
            self.num_size_cluster = dataset_config.sun_num_size_cluster
            self.mean_size_arr = dataset_config.sun_mean_size_arr
        else:
            self.num_class = dataset_config.num_class
            self.num_heading_bin = dataset_config.num_heading_bin
            self.num_size_cluster = dataset_config.num_size_cluster
            self.mean_size_arr = dataset_config.mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.dataset_config = dataset_config
        self.num_target = num_target

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)

        # self.proposal = StandardROIHeads(num_heading_bin=num_heading_bin, num_class=num_class, seed_feat_dim=256)
        self.conv1 = torch.nn.Conv1d(128, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 2 + 3 + self.num_heading_bin * 2 + self.num_size_cluster * 4 + self.num_class, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

        # Fix VoteNet
        # for _p in self.parameters():
        #     _p.requires_grad = False

        # self.use_obj_embedding = True
        # self.use_box_embedding = True
        # self.bbox_embedding = nn.Linear(27, hidden_size)
        # self.obj_embedding = nn.Linear(128, hidden_size)
        # self.features_concat = nn.Sequential(
        #     nn.Conv1d(128, hidden_size, 1),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.PReLU(hidden_size),
        #     nn.Conv1d(hidden_size, hidden_size, 1),
        # )

    def forward(self, xyz, features, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """
        self.eval()

        # Farthest point sampling (FPS) on votes
        xyz, features, sample_inds = self.vote_aggregation(xyz, features)
        data_dict['query_points_xyz'] = xyz # (batch_size, num_proposal, 3)
        data_dict['query_points_feature'] = features.permute(0, 2, 1).contiguous() # (batch_size, num_proposal, 128)
        data_dict['query_points_sample_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        # data_dict = self.proposal(features, data_dict)
        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.conv3(net)  # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)
        # data_dict = self.decode_scores(data_dict)
        # net = self.proposal(features)
        self.decode_scores(net, data_dict, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)
        # self.embed_feat(data_dict)
        # self.divide_proposals(data_dict)

        return data_dict

    def decode_pred_box(self, data_dict):
        # predicted bbox

        data_dict["pred_bbox_feature"] = data_dict["query_points_feature"]
        # data_dict["pred_bbox_mask"] = data_dict["objectness_scores"].argmax(-1)
        # data_dict["pred_bbox_sems"] = data_dict["sem_cls_scores"].argmax(-1)

        pred_center = data_dict["center"]
        pred_heading_class = torch.argmax(data_dict["heading_scores"], -1)
        pred_heading_residual = torch.gather(data_dict["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)).squeeze(2)
        pred_size_class = torch.argmax(data_dict["size_scores"], -1)
        pred_size_residual = torch.gather(data_dict["size_residuals"], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)).squeeze(2)

        pred_obb = self.dataset_config.param2obb_torch(pred_center, pred_heading_class, pred_heading_residual, pred_size_class, pred_size_residual)

        data_dict['pred_heading'] = torch.zeros_like(pred_heading_class).to(pred_heading_class.device)
        data_dict["pred_center"] = pred_center
        data_dict["pred_heading_class"] = pred_heading_class
        data_dict["pred_heading_residual"] = pred_heading_residual
        data_dict["pred_size_class"] = pred_size_class
        data_dict["pred_size_residual"] = pred_size_residual

        # batch_size, num_proposals, 8, 3
        pred_bboxes = get_3d_box_torch(pred_obb[:, :, 3:6], pred_obb[:, :, 6], pred_center[:, :, :3])
        data_dict['pred_bbox_corner'] = pred_bboxes
        data_dict["pred_size"] = pred_obb[:, :, 3:6]

    def decode_scores(self, net, data_dict, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
        """
        decode the predicted parameters for the bounding boxes

        """
        net_transposed = net.transpose(2, 1).contiguous()  # (batch_size, 1024, ..)
        batch_size = net_transposed.shape[0]
        num_proposal = net_transposed.shape[1]

        objectness_scores = net_transposed[:, :, 0:2]

        base_xyz = data_dict['aggregated_vote_xyz']  # (batch_size, num_proposal, 3)
        center = base_xyz + net_transposed[:, :, 2:5]  # (batch_size, num_proposal, 3)

        heading_scores = net_transposed[:, :, 5:5 + num_heading_bin]
        heading_residuals_normalized = net_transposed[:, :, 5 + num_heading_bin:5 + num_heading_bin * 2]

        size_scores = net_transposed[:, :, 5 + num_heading_bin * 2:5 + num_heading_bin * 2 + num_size_cluster]
        size_residuals_normalized = net_transposed[:, :,
                                    5 + num_heading_bin * 2 + num_size_cluster:5 + num_heading_bin * 2 + num_size_cluster * 4].view(
            [batch_size, num_proposal, num_size_cluster, 3])  # Bxnum_proposalxnum_size_clusterx3

        sem_cls_scores = net_transposed[:, :, 5 + num_heading_bin * 2 + num_size_cluster * 4:]  # Bxnum_proposalx18
        if hasattr(self.dataset_config, 'label_similarity'):
            sem_cls_scores = torch.matmul(sem_cls_scores.exp(), self.dataset_config.label_similarity.to(sem_cls_scores.device).transpose(0, 1)).log()

        # store
        data_dict['objectness_scores'] = objectness_scores
        data_dict['center'] = center
        data_dict['heading_scores'] = heading_scores  # Bxnum_proposalxnum_heading_bin
        data_dict[
            'heading_residuals_normalized'] = heading_residuals_normalized  # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
        data_dict['heading_residuals'] = heading_residuals_normalized * (
                    np.pi / num_heading_bin)  # Bxnum_proposalxnum_heading_bin
        data_dict['size_scores'] = size_scores
        data_dict['size_residuals_normalized'] = size_residuals_normalized
        data_dict['size_residuals'] = size_residuals_normalized * torch.from_numpy(
            mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        data_dict['sem_cls_scores'] = sem_cls_scores

        self.decode_pred_box(data_dict)

    # def embed_feat(self, data_dict):
    #     features = data_dict["pred_bbox_feature"].permute(0, 2, 1)
    #     features = self.features_concat(features).permute(0, 2, 1)
    #     batch_size, num_proposal = features.shape[:2]
    #
    #     # multiview/rgb feature embedding
    #     if self.use_obj_embedding:
    #         obj_feat = data_dict["point_clouds"][..., 6:6 + 128].permute(0, 2, 1)
    #         obj_feat_dim = obj_feat.shape[1]
    #         obj_feat_id_seed = data_dict["seed_inds"]
    #         obj_feat_id_seed = obj_feat_id_seed.long() + (
    #             (torch.arange(batch_size) * obj_feat.shape[1])[:, None].to(obj_feat_id_seed.device))
    #         obj_feat_id_seed = obj_feat_id_seed.reshape(-1)
    #         obj_feat_id_vote = data_dict["aggregated_vote_inds"]
    #         obj_feat_id_vote = obj_feat_id_vote.long() + (
    #             (torch.arange(batch_size) * data_dict["seed_inds"].shape[1])[:, None].to(
    #                 obj_feat_id_vote.device))
    #         obj_feat_id_vote = obj_feat_id_vote.reshape(-1)
    #         obj_feat_id = obj_feat_id_seed[obj_feat_id_vote]
    #         obj_feat = obj_feat.reshape(-1, obj_feat_dim)[obj_feat_id].reshape(batch_size, num_proposal,
    #                                                                            obj_feat_dim)
    #         # print(obj_feat.size())
    #         obj_embedding = self.obj_embedding(obj_feat)
    #         features = features + obj_embedding * 0.1
    #
    #     # box embedding
    #     if self.use_box_embedding:
    #         corners = data_dict['pred_bbox_corner']
    #         centers = get_bbox_centers(corners)  # batch_size, num_proposals, 3
    #         num_proposals = centers.shape[1]
    #         # attention weight
    #         manual_bbox_feat = torch.cat(
    #             [centers, (corners - centers[:, :, None, :]).reshape(batch_size, num_proposals, -1)],
    #             dim=-1).float()
    #         bbox_embedding = self.bbox_embedding(manual_bbox_feat)
    #         features = features + bbox_embedding
    #
    #     data_dict["bbox_feature"] = features



def get_bbox_centers(corners):
    coord_min = torch.min(corners, dim=2)[0] # batch_size, num_proposals, 3
    coord_max = torch.max(corners, dim=2)[0] # batch_size, num_proposals, 3
    return (coord_min + coord_max) / 2