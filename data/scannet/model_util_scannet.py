""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/scannet/model_util_scannet.py
"""

import numpy as np
import sys
import os
import pickle
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.getcwd(), os.pardir, "lib")) # HACK add the lib folder

from lib.configs.config import CONF

sys.path.append(os.path.join(os.getcwd(), os.pardir, "utils")) # HACK add the lib folder
from box_util import get_3d_box

GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def rotate_aligned_boxes(input_boxes, rot_mat):    
    centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]    
    new_centers = np.dot(centers, np.transpose(rot_mat))
           
    dx, dy = lengths[:,0]/2.0, lengths[:,1]/2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))
    
    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):        
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:,0] = crnr[0]*dx
        crnrs[:,1] = crnr[1]*dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:,i] = crnrs[:,0]
        new_y[:,i] = crnrs[:,1]
    
    
    new_dx = 2.0*np.max(new_x, 1)
    new_dy = 2.0*np.max(new_y, 1)    
    new_lengths = np.stack((new_dx, new_dy, lengths[:,2]), axis=1)
                  
    return np.concatenate([new_centers, new_lengths], axis=1)

def rotate_aligned_boxes_along_axis(input_boxes, rot_mat, axis):    
    centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]    
    new_centers = np.dot(centers, np.transpose(rot_mat))

    if axis == "x":     
        d1, d2 = lengths[:,1]/2.0, lengths[:,2]/2.0
    elif axis == "y":
        d1, d2 = lengths[:,0]/2.0, lengths[:,2]/2.0
    else:
        d1, d2 = lengths[:,0]/2.0, lengths[:,1]/2.0

    new_1 = np.zeros((d1.shape[0], 4))
    new_2 = np.zeros((d1.shape[0], 4))
    
    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):        
        crnrs = np.zeros((d1.shape[0], 3))
        crnrs[:,0] = crnr[0]*d1
        crnrs[:,1] = crnr[1]*d2
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_1[:,i] = crnrs[:,0]
        new_2[:,i] = crnrs[:,1]
    
    new_d1 = 2.0*np.max(new_1, 1)
    new_d2 = 2.0*np.max(new_2, 1)    

    if axis == "x":     
        new_lengths = np.stack((lengths[:,0], new_d1, new_d2), axis=1)
    elif axis == "y":
        new_lengths = np.stack((new_d1, lengths[:,1], new_d2), axis=1)
    else:
        new_lengths = np.stack((new_d1, new_d2, lengths[:,2]), axis=1)
                  
    return np.concatenate([new_centers, new_lengths], axis=1)


class ScannetDatasetConfig(object):
    def __init__(self):
        self.type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}  
        self.class2type = {self.type2class[t]:t for t in self.type2class}

        self.nyu40ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)
        self.nyu40id2class = self._get_nyu40id2class()
        self.mean_size_arr = np.load(os.path.join(CONF.PATH.SCANNET, 'meta_data/scannet_reference_means.npz'))['arr_0']
        # print(self.mean_size_arr)
        self.num_class = len(self.type2class.keys())
        self.num_heading_bin = 1
        self.num_size_cluster = len(self.type2class.keys())

        self.type_mean_size = {}
        for i in range(self.num_size_cluster):
            self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i,:]

    def _get_nyu40id2class(self):
        lines = [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET, 'meta_data/scannetv2-labels.combined.tsv'))]
        lines = lines[1:]
        nyu40ids2class = {}
        for i in range(len(lines)):
            label_classes_set = set(self.type2class.keys())
            elements = lines[i].split('\t')
            nyu40_id = int(elements[4])
            nyu40_name = elements[7]
            if nyu40_id in self.nyu40ids:
                if nyu40_name not in label_classes_set:
                    nyu40ids2class[nyu40_id] = self.type2class["others"]
                else:
                    nyu40ids2class[nyu40_id] = self.type2class[nyu40_name]

        return nyu40ids2class

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle

            NOT USED.
        '''
        assert(False)
    
    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.
        
        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return 0

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.
        
        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return np.zeros(pred_cls.shape[0])

    def class2angle_torch(self, pred_cls, residual):
        return torch.zeros_like(residual).to(residual.device)

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual
    
    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''      
        return self.mean_size_arr[pred_cls] + residual

    def class2size_batch(self, pred_cls, residual):
        ''' Inverse function to size2class '''      
        return self.mean_size_arr[pred_cls] + residual

    def class2size_torch(self, pred_cls, residual):
        mean_size_arr = torch.Tensor(self.mean_size_arr).to(pred_cls.device)
        return mean_size_arr[pred_cls] + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb

    def param2obb_batch(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle_batch(heading_class, heading_residual)
        box_size = self.class2size_batch(size_class, size_residual)
        obb = np.zeros((heading_class.shape[0], 7))
        obb[:, 0:3] = center
        obb[:, 3:6] = box_size
        obb[:, 6] = heading_angle*-1
        return obb

    def param2obb_torch(self, center, heading_class, heading_residual, size_class, size_residual):
        # center, size_residual B * N * 3
        # heading_class, heading_residual, size_class B * N
        heading_angle = self.class2angle_torch(heading_class, heading_residual)
        box_size = self.class2size_torch(size_class, size_residual)
        obb = torch.zeros((tuple(size_class.size()) + (7,))).to(center.device)
        # print(obb.size())
        obb[:, :, :3] = center
        obb[:, :, 3:6] = box_size
        obb[:, :, 6] = heading_angle * -1
        return obb


def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class SunToScannetDatasetConfig(object):
    def __init__(self):
        # scannet
        self.type2class = {'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
                           'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
                           'refrigerator': 12, 'shower curtain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16,
                           'garbage bin': 17}
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.num_class = len(self.type2class)
        self.num_heading_bin = 1
        self.num_size_cluster = len(self.type2class)
        self.mean_size_arr = np.load(os.path.join(CONF.PATH.SCANNET, 'meta_data/scannet_reference_means.npz'))['arr_0']
        # a = np.load(os.path.join(CONF.PATH.SCANNET, 'meta_data/scannet_reference_means.npz'))

        # sunrgbd
        self.sun_type_mean_size = {'bathtub': np.array([0.765840, 1.398258, 0.472728]),
                                   'bed': np.array([2.114256, 1.620300, 0.927272]),
                                   'bookshelf': np.array([0.404671, 1.071108, 1.688889]),
                                   'chair': np.array([0.591958, 0.552978, 0.827272]),
                                   'desk': np.array([0.695190, 1.346299, 0.736364]),
                                   'dresser': np.array([0.528526, 1.002642, 1.172878]),
                                   'night_stand': np.array([0.500618, 0.632163, 0.683424]),
                                   'sofa': np.array([0.923508, 1.867419, 0.845495]),
                                   'table': np.array([0.791118, 1.279516, 0.718182]),
                                   'toilet': np.array([0.699104, 0.454178, 0.756250])}
        self.sun_num_class = len(self.sun_type_mean_size)
        self.sun_type2class = {'bed': 0, 'table': 1, 'sofa': 2, 'chair': 3, 'toilet': 4, 'desk': 5, 'dresser': 6,
                               'night_stand': 7, 'bookshelf': 8, 'bathtub': 9}
        self.sun_class2type = {self.sun_type2class[t]: t for t in self.sun_type2class}
        self.sun_num_heading_bin = 12
        self.sun_num_size_cluster = self.sun_num_class
        self.sun_mean_size_arr = np.zeros((self.sun_num_class, 3))
        for i in range(self.sun_num_class):
            self.sun_mean_size_arr[i, :] = self.sun_type_mean_size[self.sun_class2type[i]]

        # sim matrix
        glove = pickle.load(open(GLOVE_PICKLE, "rb"))
        self.scan_label_feat = np.zeros((self.num_class, 300))
        self.sun_label_feat = np.zeros((self.sun_num_class, 300))
        for i in range(self.num_class):
            if self.class2type[i] == "shower curtain":
                self.scan_label_feat[i] = (glove["shower"] + glove["curtain"]) / 2.
            elif self.class2type[i] == "garbage bin":
                self.scan_label_feat[i] = (glove["garbage"] + glove["bin"]) / 2.
            else:
                self.scan_label_feat[i] = glove[self.class2type[i]]
        for i in range(self.sun_num_class):
            if self.sun_class2type[i] == "night_stand":
                self.sun_label_feat[i] = glove["nightstand"]
            else:
                self.sun_label_feat[i] = glove[self.sun_class2type[i]]
        print(self.scan_label_feat.shape, self.sun_label_feat.shape)
        # print(self.class2type, self.sun_class2type)
        self.scan_label_feat = torch.from_numpy(self.scan_label_feat.astype(np.float32))
        self.sun_label_feat = torch.from_numpy(self.sun_label_feat.astype(np.float32))
        self.label_similarity = sim_matrix(self.scan_label_feat, self.sun_label_feat)
        # print(self.label_similarity)
        for i in range(self.label_similarity.size(0)):
            flag = -1
            for j in range(self.label_similarity.size(1)):
                if self.label_similarity[i][j] >= 1.0 - 1e-8:
                    flag = j
            # print(i, flag)
            if flag != -1:
                self.label_similarity[i].clamp_max_(0.)
                self.label_similarity[i][flag] = 1.
        self.label_similarity = F.normalize(self.label_similarity.clamp_min(0.), p=1, dim=1)

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.

        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return 0

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.

        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return np.zeros(pred_cls.shape[0])

    def class2angle_torch(self, pred_cls, residual):
        return torch.zeros_like(residual).to(residual.device)

    # def size2class(self, size, type_name):
    #     ''' Convert 3D box size (l,w,h) to size class and size residual '''
    #     size_class = self.type2class[type_name]
    #     size_residual = size - self.type_mean_size[type_name]
    #     return size_class, size_residual

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        return self.mean_size_arr[pred_cls] + residual

    def class2size_batch(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        return self.mean_size_arr[pred_cls] + residual

    def class2size_torch(self, pred_cls, residual):
        mean_size_arr = torch.Tensor(self.sun_mean_size_arr).to(pred_cls.device)
        return mean_size_arr[pred_cls] + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

    def param2obb_batch(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle_batch(heading_class, heading_residual)
        box_size = self.class2size_batch(size_class, size_residual)
        obb = np.zeros((heading_class.shape[0], 7))
        obb[:, 0:3] = center
        obb[:, 3:6] = box_size
        obb[:, 6] = heading_angle * -1
        return obb

    def param2obb_torch(self, center, heading_class, heading_residual, size_class, size_residual):
        # center, size_residual B * N * 3
        # heading_class, heading_residual, size_class B * N
        heading_angle = self.class2angle_torch(heading_class, heading_residual)
        box_size = self.class2size_torch(size_class, size_residual)
        obb = torch.zeros((tuple(size_class.size()) + (7,))).to(center.device)
        # print(obb.size())
        obb[:, :, :3] = center
        obb[:, :, 3:6] = box_size
        obb[:, :, 6] = heading_angle * -1
        return obb
