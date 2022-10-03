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
    def __init__(self, num_proposal, sampling, seed_feat_dim=256, dataset_config=None):
        super().__init__()
        # num_class = 10
        # num_heading_bin = 12
        # num_size_cluster = 10
        # self.type2class = {'bed': 0, 'table': 1, 'sofa': 2, 'chair': 3, 'toilet': 4, 'desk': 5, 'dresser': 6,
        #                    'night_stand': 7, 'bookshelf': 8, 'bathtub': 9}
        # self.class2type = {self.type2class[t]: t for t in self.type2class}
        # self.type_mean_size = {'bathtub': np.array([0.765840, 1.398258, 0.472728]),
        #                        'bed': np.array([2.114256, 1.620300, 0.927272]),
        #                        'bookshelf': np.array([0.404671, 1.071108, 1.688889]),
        #                        'chair': np.array([0.591958, 0.552978, 0.827272]),
        #                        'desk': np.array([0.695190, 1.346299, 0.736364]),
        #                        'dresser': np.array([0.528526, 1.002642, 1.172878]),
        #                        'night_stand': np.array([0.500618, 0.632163, 0.683424]),
        #                        'sofa': np.array([0.923508, 1.867419, 0.845495]),
        #                        'table': np.array([0.791118, 1.279516, 0.718182]),
        #                        'toilet': np.array([0.699104, 0.454178, 0.756250])}
        # mean_size_arr = np.zeros((num_size_cluster, 3))
        # for i in range(num_size_cluster):
        #     mean_size_arr[i, :] = self.type_mean_size[self.class2type[i]]
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
        xyz, features, fps_inds = self.vote_aggregation(xyz, features)

        sample_inds = fps_inds

        data_dict['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        data_dict['aggregated_vote_features'] = features.permute(0, 2, 1).contiguous() # (batch_size, num_proposal, 128)
        data_dict['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        # data_dict = self.proposal(features, data_dict)
        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.conv3(net)  # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)
        # data_dict = self.decode_scores(data_dict)
        # net = self.proposal(features)
        data_dict = self.decode_scores(net, data_dict, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)

        return data_dict

    def decode_pred_box(self, data_dict):
        # predicted bbox

        data_dict["pred_bbox_feature"] = data_dict["aggregated_vote_features"]
        # data_dict["pred_bbox_mask"] = data_dict["objectness_scores"].argmax(-1)
        # data_dict["pred_bbox_sems"] = data_dict["sem_cls_scores"].argmax(-1)

        # aggregated_vote_xyz = data_dict["aggregated_vote_xyz"] # (B,K,3)
        # pred_heading_class = torch.argmax(data_dict["heading_scores"], -1) # B,num_proposal
        # pred_heading_residual = torch.gather(data_dict["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1


        # pred_box_size = rois[:, :, 0:3] + rois[:, :, 3:6]  # (B, N, 3)
        # data_dict['pred_size'] = pred_box_size

        # bsize = pred_box_size.shape[0]
        # num_proposal = pred_box_size.shape[1]

        # Compute pred center xyz
        # vote_xyz = (rois[:,:,0:3] - rois[:,:,3:6]) / 2  # (B, N, 3)
        # R = rotz_batch_pytorch(pred_heading.float()).view(-1, 3, 3)
        # R = roty_batch_pytorch(pred_heading.float()).view(-1, 3, 3)
        # vote_xyz = torch.matmul(vote_xyz.reshape(-1, 3).unsqueeze(1), R).squeeze(2)  # (B, N, 3)
        # vote_xyz = vote_xyz.view(bsize, num_proposal, 3)
        # pred_center = aggregated_vote_xyz - vote_xyz
        pred_center = data_dict["center"]
        pred_heading_class = torch.argmax(data_dict["heading_scores"], -1)
        pred_heading_residual = torch.gather(data_dict["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)).squeeze(2)
        pred_size_class = torch.argmax(data_dict["size_scores"], -1)
        pred_size_residual = torch.gather(data_dict["size_residuals"], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)).squeeze(2)
        # print(pred_center.size())
        # print(pred_heading_class.size(), pred_heading_residual.size())
        # print(pred_size_class.size(), pred_size_residual.size())

        pred_obb = self.dataset_config.param2obb_torch(pred_center, pred_heading_class, pred_heading_residual, pred_size_class, pred_size_residual)

        # pred_heading = pred_heading_class.float() * (2.0 * np.pi / self.num_heading_bin) + pred_heading_residual[..., 0]
        data_dict['pred_heading'] = torch.zeros_like(pred_heading_class).to(pred_heading_class.device)
        data_dict["pred_center"] = pred_center
        data_dict["pred_heading_class"] = pred_heading_class
        data_dict["pred_heading_residual"] = pred_heading_residual
        data_dict["pred_size_class"] = pred_size_class
        data_dict["pred_size_residual"] = pred_size_residual

        # batch_size, num_proposals, 8, 3
        pred_bboxes = get_3d_box_torch(pred_obb[:, :, 3:6], pred_obb[:, :, 6], pred_center[:, :, :3])
        # pred_bboxes = torch.from_numpy(pred_bboxes).float().to(pred_center.device)# .reshape(bsize, num_proposal, 8, 3)
        data_dict['pred_bbox_corner'] = pred_bboxes
        data_dict["pred_size"] = pred_obb[:, :, 3:6]

        # Testing Scripts
        # center = pred_center[0, 13].cpu().numpy()
        # heading = pred_heading[0, 13].cpu().numpy()
        # size = pred_box_size[0, 13].cpu().numpy()
        # from utils.box_util import get_3d_box
        # newbox = get_3d_box(size, heading, center)

        # print(newbox, pred_bboxes[0, 13])
        # import ipdb
        # ipdb.set_trace()
        # # print(pred_center, pred_size)
        return data_dict

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

        sem_cls_scores = net_transposed[:, :, 5 + num_heading_bin * 2 + num_size_cluster * 4:]  # Bxnum_proposalx10
        # print(sem_cls_scores.size(), sem_cls_scores[0][0], sem_cls_scores.exp().log()[0][0])
        # print(sem_cls_scores.dtype, self.dataset_config.label_similarity.dtype)
        if hasattr(self.dataset_config, 'label_similarity'):
            sem_cls_scores = torch.matmul(sem_cls_scores.exp(), self.dataset_config.label_similarity.to(sem_cls_scores.device).transpose(0, 1)).log()
        # print(sem_cls_scores.size(), sem_cls_scores[0][0])

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

        return data_dict
