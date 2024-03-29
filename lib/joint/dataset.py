'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import h5py
import json
import pickle
import random
import numpy as np
import multiprocessing as mp
import torch

from itertools import chain
from collections import Counter
from torch.utils.data import Dataset

from config.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from utils.box_util import get_3d_box, get_3d_box_batch
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
OBJ_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
# SCANREFER_VOCAB = os.path.join(CONF.PATH.DATA, "ScanRefer_vocabulary.json")
VOCAB = os.path.join(CONF.PATH.DATA, "{}_vocabulary.json") # dataset_name
# SCANREFER_VOCAB_WEIGHTS = os.path.join(CONF.PATH.DATA, "ScanRefer_vocabulary_weights.json")
VOCAB_WEIGHTS = os.path.join(CONF.PATH.DATA, "{}_vocabulary_weights.json") # dataset_name
# MULTIVIEW_DATA = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")

class ReferenceDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            # print(lines[i])
            # print(raw_name, nyu40_name)
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]
        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _tranform_des(self):
        lang = {}
        label = {}
        lang_main = {}
        label_main = {}

        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]
            object_name = " ".join(data["object_name"].split("_"))

            if scene_id not in lang:
                lang[scene_id] = {}
                label[scene_id] = {}
                lang_main[scene_id] = {}
                label_main[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}
                label[scene_id][object_id] = {}
                lang_main[scene_id][object_id] = {}
                label_main[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}
                label[scene_id][object_id][ann_id] = {}
                lang_main[scene_id][object_id][ann_id] = {}
                lang_main[scene_id][object_id][ann_id]["main"] = {}
                lang_main[scene_id][object_id][ann_id]["len"] = 0
                lang_main[scene_id][object_id][ann_id]["first_obj"] = -1
                lang_main[scene_id][object_id][ann_id]["unk"] = self.glove["unk"]

            # trim long descriptions
            tokens = data["token"][:CONF.TRAIN.MAX_DES_LEN]

            # tokenize the description
            tokens = ["sos"] + tokens + ["eos"]
            embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN + 2, 300))
            labels = np.zeros((CONF.TRAIN.MAX_DES_LEN + 2)) # start and end
            main_embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN + 2, 300))
            pd = 1

            main_object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17
            # load
            for token_id in range(len(tokens)):
                token = tokens[token_id]
                try:
                    embeddings[token_id] = self.glove[token]
                    labels[token_id] = self.vocabulary["word2idx"][token]
                except KeyError:
                    embeddings[token_id] = self.glove["unk"]
                    labels[token_id] = self.vocabulary["word2idx"]["unk"]
                if pd == 1:
                    try:
                        main_embeddings[token_id] = self.glove[token]
                    except KeyError:
                        main_embeddings[token_id] = self.glove["unk"]
                    if token == ".":
                        pd = 0
                        main_embeddings[token_id+1] = self.glove["eos"]
                        lang_main[scene_id][object_id][ann_id]["len"] = token_id + 2
                object_cat = self.raw2label[token] if token in self.raw2label else -1
                is_two_words = 0
                if token_id + 1 < len(tokens):
                    token_new = token + " " + tokens[token_id + 1]
                    object_cat_new = self.raw2label[token_new] if token_new in self.raw2label else -1
                    if object_cat_new != -1:
                        object_cat = object_cat_new
                        is_two_words = 1
                if lang_main[scene_id][object_id][ann_id]["first_obj"] == -1 and object_cat == main_object_cat:
                    if is_two_words == 1 and token_id + 1 < len(tokens):
                        lang_main[scene_id][object_id][ann_id]["first_obj"] = token_id + 1
                    else:
                        lang_main[scene_id][object_id][ann_id]["first_obj"] = token_id

            if pd == 1:
                lang_main[scene_id][object_id][ann_id]["len"] = len(tokens)
            
            # store
            lang[scene_id][object_id][ann_id] = embeddings
            label[scene_id][object_id][ann_id] = labels
            lang_main[scene_id][object_id][ann_id]["main"] = main_embeddings

        return lang, label, lang_main

    def _ground_tranform_des(self):
        with open(GLOVE_PICKLE, "rb") as f:
            glove = pickle.load(f)

        lang = {}
        label = {}
        lang_main = {}
        masks = {}

        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]
            object_name = " ".join(data["object_name"].split("_"))

            if "mask_list" in data:
                # attribute_mask_list = data["mask_first_list"][:CONF.TRAIN.MAX_GROUND_DES_LEN]
                relation_mask_list = data["mask_list"][:CONF.TRAIN.MAX_GROUND_DES_LEN]
                # attribute_mask_list = attribute_mask_list
                # relation_mask_list = relation_mask_list
                # attribute_mask_num = torch.tensor(attribute_mask_list).sum()
                relation_mask_num = torch.tensor(relation_mask_list).sum()

                all_mask_list_ = relation_mask_list
                all_mask_num = relation_mask_num

                if "mask_first_list" in data:
                    attribute_mask_list = data["mask_first_list"][:CONF.TRAIN.MAX_GROUND_DES_LEN]
                    attribute_mask_list = attribute_mask_list
                    attribute_mask_num = torch.tensor(attribute_mask_list).sum()
                    for i in range(len(attribute_mask_list)):
                        all_mask_list_[i] = all_mask_list_[i] + attribute_mask_list[i]
                    all_mask_num += attribute_mask_num

                # attr_mask_list = np.ones(CONF.TRAIN.MAX_GROUND_DES_LEN) * 2
                # rel_mask_list = np.ones(CONF.TRAIN.MAX_GROUND_DES_LEN) * 2
                all_mask_list = np.ones(CONF.TRAIN.MAX_GROUND_DES_LEN) * 2
                # for i in range(len(attribute_mask_list)):
                #     attr_mask_list[i] = attribute_mask_list[i]
                # for i in range(len(relation_mask_list)):
                #     rel_mask_list[i] = relation_mask_list[i]
                for i in range(len(all_mask_list_)):
                    all_mask_list[i] = all_mask_list_[i]  # 0: words,  1: masked_words,  2: padding
            else:
                # attr_mask_list = np.zeros(CONF.TRAIN.MAX_GROUND_DES_LEN)
                # rel_mask_list = np.zeros(CONF.TRAIN.MAX_GROUND_DES_LEN)
                all_mask_list = np.zeros(CONF.TRAIN.MAX_GROUND_DES_LEN)
                # attribute_mask_num = torch.tensor(attr_mask_list).sum()
                # relation_mask_num = torch.tensor(rel_mask_list).sum()
                all_mask_num = torch.tensor(all_mask_list).sum()

            if scene_id not in lang:
                lang[scene_id] = {}
                label[scene_id] = {}
                lang_main[scene_id] = {}
                masks[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}
                label[scene_id][object_id] = {}
                lang_main[scene_id][object_id] = {}
                masks[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}
                label[scene_id][object_id][ann_id] = {}
                lang_main[scene_id][object_id][ann_id] = {}
                lang_main[scene_id][object_id][ann_id]["main"] = {}
                lang_main[scene_id][object_id][ann_id]["len"] = 0
                lang_main[scene_id][object_id][ann_id]["first_obj"] = -1
                lang_main[scene_id][object_id][ann_id]["unk"] = glove["unk"]

                masks[scene_id][object_id][ann_id] = {}
                # masks[scene_id][object_id][ann_id]['attr_masks'] = {}
                # masks[scene_id][object_id][ann_id]['attr_masks_num'] = {}
                # masks[scene_id][object_id][ann_id]['rel_masks'] = {}
                # masks[scene_id][object_id][ann_id]['rel_masks_num'] = {}
                masks[scene_id][object_id][ann_id]['all_masks'] = {}
                masks[scene_id][object_id][ann_id]['all_masks_num'] = {}

            # tokenize the description
            tokens = data["token"]
            embeddings = np.zeros((CONF.TRAIN.MAX_GROUND_DES_LEN, 300))
            labels = np.zeros(CONF.TRAIN.MAX_GROUND_DES_LEN)
            main_embeddings = np.zeros((CONF.TRAIN.MAX_GROUND_DES_LEN, 300))
            pd = 1
            # if object_name[:6] == "shower":
            #     print(object_name, self.raw2label[object_name] if object_name in self.raw2label else 17)
            main_object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17
            # if main_object_cat == 17:
            #     print(object_name)
            for token_id in range(CONF.TRAIN.MAX_GROUND_DES_LEN):
                if token_id < len(tokens):
                    token = tokens[token_id]
                    if token in glove:
                        embeddings[token_id] = glove[token]
                        labels[token_id] = self.vocabulary["word2idx"][token]
                    else:
                        embeddings[token_id] = glove["pad"]
                        labels[token_id] = self.vocabulary["word2idx"]["pad_"]
                    if pd == 1:
                        if token in glove:
                            main_embeddings[token_id] = glove[token]
                        else:
                            main_embeddings[token_id] = glove["unk"]
                        if token == ".":
                            pd = 0
                            lang_main[scene_id][object_id][ann_id]["len"] = token_id + 1
                    object_cat = self.raw2label[token] if token in self.raw2label else -1
                    is_two_words = 0
                    if token_id + 1 < len(tokens):
                        token_new = token + " " + tokens[token_id+1]
                        object_cat_new = self.raw2label[token_new] if token_new in self.raw2label else -1
                        if object_cat_new != -1:
                            object_cat = object_cat_new
                            is_two_words = 1
                    # if object_name == "shower_curtain" and object_cat == 13:
                    #     print(object_name, object_cat)
                    if lang_main[scene_id][object_id][ann_id]["first_obj"] == -1 and object_cat == main_object_cat:
                        if is_two_words == 1 and token_id + 1 < len(tokens):
                            lang_main[scene_id][object_id][ann_id]["first_obj"] = token_id + 1
                        else:
                            lang_main[scene_id][object_id][ann_id]["first_obj"] = token_id
            if pd == 1:
                lang_main[scene_id][object_id][ann_id]["len"] = len(tokens)

            # store
            lang[scene_id][object_id][ann_id] = embeddings
            label[scene_id][object_id][ann_id] = labels
            lang_main[scene_id][object_id][ann_id]["main"] = main_embeddings

            masks[scene_id][object_id][ann_id]['all_masks'] = all_mask_list
            masks[scene_id][object_id][ann_id]['all_masks_num'] = all_mask_num

        return lang, label, lang_main, masks

    def _build_vocabulary(self, dataset_name):
        vocab_path = VOCAB.format(dataset_name)
        if os.path.exists(vocab_path):
            self.vocabulary = json.load(open(vocab_path))
            all_words = chain(*[data["token"][:CONF.TRAIN.MAX_GROUND_DES_LEN] for data in self.scanrefer])
            word_counter = Counter(all_words)
            word_counter = sorted([(k, v) for k, v in word_counter.items() if k in self.glove], key=lambda x: x[1],
                                  reverse=True)
            word_list = [k for k, _ in word_counter]
            word2idx, idx2word = self.vocabulary["word2idx"], self.vocabulary["idx2word"]
            new_list = []
            for w in word_list:
                if w not in word2idx.keys():
                    new_list.append(w)
            shift = len(word2idx)
            for i, w in enumerate(new_list):
                shifted_i = i + shift
                word2idx[w] = shifted_i
                idx2word[shifted_i] = w
            self.vocabulary = {
                "word2idx": word2idx,
                "idx2word": idx2word
            }
            json.dump(self.vocabulary, open(vocab_path, "w"), indent=4)
        else:
            all_words = chain(*[data["token"][:CONF.TRAIN.MAX_GROUND_DES_LEN] for data in self.scanrefer])
            word_counter = Counter(all_words)
            word_counter = sorted([(k, v) for k, v in word_counter.items() if k in self.glove], key=lambda x: x[1], reverse=True)
            word_list = [k for k, _ in word_counter]

            # build vocabulary
            word2idx, idx2word = {}, {}
            spw = ["pad_", "unk", "sos", "eos"] # NOTE distinguish padding token "pad_" and the actual word "pad"
            # add special words into vocabulary
            if "pad_" not in word2idx.keys():
                for i, w in enumerate(spw):
                    word2idx[w] = i
                    idx2word[i] = w

            for i, w in enumerate(word_list):
                shifted_i = i + len(spw)
                word2idx[w] = shifted_i
                idx2word[shifted_i] = w

            vocab = {
                "word2idx": word2idx,
                "idx2word": idx2word
            }
            json.dump(vocab, open(vocab_path, "w"), indent=4)
            self.vocabulary = vocab

    def _build_frequency(self, dataset_name):
        vocab_weights_path = VOCAB_WEIGHTS.format(dataset_name)
        if os.path.exists(vocab_weights_path):
            with open(vocab_weights_path) as f:
                weights = json.load(f)
                self.weights = np.array([v for _, v in weights.items()])
        else:
            all_tokens = []
            for scene_id in self.lang_ids.keys():
                for object_id in self.lang_ids[scene_id].keys():
                    for ann_id in self.lang_ids[scene_id][object_id].keys():
                        all_tokens += self.lang_ids[scene_id][object_id][ann_id].astype(int).tolist()

            word_count = Counter(all_tokens)
            word_count = sorted([(k, v) for k, v in word_count.items()], key=lambda x: x[0])
            
            # frequencies = [c for _, c in word_count]
            # weights = np.array(frequencies).astype(float)
            # weights = weights / np.sum(weights)
            # weights = 1 / np.log(1.05 + weights)

            weights = np.ones((len(word_count)))

            self.weights = weights
            
            with open(vocab_weights_path, "w") as f:
                weights = {k: v for k, v in enumerate(weights)}
                json.dump(weights, f, indent=4)

    def _load_data(self, dataset_name):
        print("loading data...")
        # load language features
        self.glove = pickle.load(open(GLOVE_PICKLE, "rb"))
        self._build_vocabulary(dataset_name)
        self.num_vocabs = len(self.vocabulary["word2idx"].keys())
        self.raw2label = self._get_raw2label()
        self.lang, self.lang_ids, self.lang_main = self._tranform_des()
        self.ground_lang, self.ground_lang_ids, self.ground_lang_main, self.masks = self._ground_tranform_des()
        self._build_frequency(dataset_name)

        if not CONF.pretrain_model_on:
            pretrained_path = CONF.PATH.PRETRAINED_TRAIN_DATA if self.split == "train" else CONF.PATH.PRETRAINED_VAL_DATA
            self.pretrained_data = torch.load(pretrained_path)

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            # self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_vert.npy")
            self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy")
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
            # self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_bbox.npy")
            self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_bbox.npy")

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid

        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox


class ScannetReferenceDataset(ReferenceDataset):
       
    def __init__(self, scanrefer, scanrefer_new, scanrefer_all_scene,
        split="train", 
        name="ScanRefer",
        num_points=40000,
        lang_num_max=8,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False, 
        augment=False,
        shuffle=False,
        scan2cad_rotation=None):

        # NOTE only feed the scan2cad_rotation when on the training mode and train split

        self.scanrefer = scanrefer
        self.scanrefer_new = scanrefer_new
        self.scanrefer_new_len = len(scanrefer_new)
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.name = name
        self.num_points = num_points
        self.lang_num_max = lang_num_max
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment
        self.scan2cad_rotation = scan2cad_rotation

        # load data
        self._load_data(name)
        self.multiview_data = {}
        self.should_shuffle = shuffle
       
    def __len__(self):
        #return len(self.scanrefer)
        return self.scanrefer_new_len

    def split_scene_new(self,  scanrefer_data):
        scanrefer_train_new = []
        scanrefer_train_new_scene, scanrefer_train_scene = [], []
        scene_id = ''
        lang_num_max = self.lang_num_max
        for data in scanrefer_data:
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(scanrefer_train_scene) > 0:
                    if self.should_shuffle:
                        random.shuffle(scanrefer_train_scene)
                    # print("scanrefer_train_scene", len(scanrefer_train_scene))
                    for new_data in scanrefer_train_scene:
                        if len(scanrefer_train_new_scene) >= lang_num_max:
                            scanrefer_train_new.append(scanrefer_train_new_scene)
                            scanrefer_train_new_scene = []
                        scanrefer_train_new_scene.append(new_data)
                    if len(scanrefer_train_new_scene) > 0:
                        scanrefer_train_new.append(scanrefer_train_new_scene)
                        scanrefer_train_new_scene = []
                    scanrefer_train_scene = []
            scanrefer_train_scene.append(data)
        if len(scanrefer_train_scene) > 0:
            if self.should_shuffle:
                random.shuffle(scanrefer_train_scene)
            # print("scanrefer_train_scene", len(scanrefer_train_scene))
            for new_data in scanrefer_train_scene:
                if len(scanrefer_train_new_scene) >= lang_num_max:
                    scanrefer_train_new.append(scanrefer_train_new_scene)
                    scanrefer_train_new_scene = []
                scanrefer_train_new_scene.append(new_data)
            if len(scanrefer_train_new_scene) > 0:
                scanrefer_train_new.append(scanrefer_train_new_scene)
                scanrefer_train_new_scene = []
        return scanrefer_train_new

    def shuffle_data(self):
        # if self.shuffled:
        #     return
        # SElf.shuffled = True
        print('shuffle dataset data(lang)', flush=True)
        self.scanrefer_new = self.split_scene_new(self.scanrefer)
        if self.should_shuffle:
            random.shuffle(self.scanrefer_new)
        assert len(self.scanrefer_new) == self.scanrefer_new_len, 'assert scanrefer length right'
        print('shuffle done', flush=True)

    def __getitem__(self, idx):
        start = time.time()
        lang_num = len(self.scanrefer_new[idx])
        scene_id = self.scanrefer_new[idx][0]["scene_id"]

        object_id_list = []
        object_name_list = []
        ann_id_list = []
        easy_hard_list = []
        dep_indep_list = []

        lang_feat_list = []
        lang_len_list = []
        lang_ids_list = []
        main_lang_feat_list = []
        main_lang_len_list = []
        first_obj_list = []
        unk_list = []

        ground_lang_feat_list = []
        ground_lang_len_list = []
        ground_lang_ids_list = []
        ground_main_lang_feat_list = []
        ground_main_lang_len_list = []
        ground_first_obj_list = []

        # attr_masks_list = []
        # attr_masks_num_list = []
        # rel_masks_list = []
        # rel_masks_num_list = []
        all_masks_list = []
        all_masks_num_list = []

        for i in range(self.lang_num_max):
            if i < lang_num:
                object_id = int(self.scanrefer_new[idx][i]["object_id"])
                object_name = " ".join(self.scanrefer_new[idx][i]["object_name"].split("_"))
                ann_id = self.scanrefer_new[idx][i]["ann_id"]
                if self.name != "ScanRefer":
                    easy_hard = 0 if int(self.scanrefer_new[idx][i]["hardness"]) <= 2 else 1
                else:
                    easy_hard = 0
                if self.name != "ScanRefer":
                    target_words = {'front', 'behind', 'back', 'right', 'left', 'facing', 'leftmost', 'rightmost',
                                    'looking', 'across'}
                    tokens = self.scanrefer_new[idx][i]["token"]
                    dep_indep = 0 if len(set(tokens).intersection(target_words)) > 0 else 1
                else:
                    dep_indep = 0

                lang_feat = self.lang[scene_id][str(object_id)][ann_id]
                lang_len = len(self.scanrefer_new[idx][i]["token"]) + 2
                lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN + 2 else CONF.TRAIN.MAX_DES_LEN + 2
                lang_ids = self.lang_ids[scene_id][str(object_id)][ann_id]
                main_lang_feat = self.lang_main[scene_id][str(object_id)][ann_id]["main"]
                main_lang_len = self.lang_main[scene_id][str(object_id)][ann_id]["len"]
                first_obj = self.lang_main[scene_id][str(object_id)][ann_id]["first_obj"]
                unk = self.lang_main[scene_id][str(object_id)][ann_id]["unk"]

                ground_lang_feat = self.ground_lang[scene_id][str(object_id)][ann_id]
                ground_lang_len = len(self.scanrefer_new[idx][i]["token"])
                ground_lang_len = ground_lang_len if ground_lang_len <= CONF.TRAIN.MAX_GROUND_DES_LEN else CONF.TRAIN.MAX_GROUND_DES_LEN
                ground_lang_ids = self.ground_lang_ids[scene_id][str(object_id)][ann_id]
                ground_main_lang_feat = self.ground_lang_main[scene_id][str(object_id)][ann_id]["main"]
                ground_main_lang_len = self.ground_lang_main[scene_id][str(object_id)][ann_id]["len"]
                ground_first_obj = self.ground_lang_main[scene_id][str(object_id)][ann_id]["first_obj"]

                # attr_masks = self.masks[scene_id][str(object_id)][ann_id]["attr_masks"]
                # attr_masks_num = self.masks[scene_id][str(object_id)][ann_id]['attr_masks_num']
                # rel_masks = self.masks[scene_id][str(object_id)][ann_id]['rel_masks']
                # rel_masks_num = self.masks[scene_id][str(object_id)][ann_id]['rel_masks_num']
                all_masks = self.masks[scene_id][str(object_id)][ann_id]["all_masks"]
                all_masks_num = self.masks[scene_id][str(object_id)][ann_id]['all_masks_num']

            object_id_list.append(object_id)
            object_name_list.append(object_name)
            ann_id_list.append(ann_id)
            easy_hard_list.append(easy_hard)
            dep_indep_list.append(dep_indep)

            lang_feat_list.append(lang_feat)
            lang_len_list.append(lang_len)
            lang_ids_list.append(lang_ids)
            main_lang_feat_list.append(main_lang_feat)
            main_lang_len_list.append(main_lang_len)
            first_obj_list.append(first_obj)
            unk_list.append(unk)

            ground_lang_feat_list.append(ground_lang_feat)
            ground_lang_len_list.append(ground_lang_len)
            ground_lang_ids_list.append(ground_lang_ids)
            ground_main_lang_feat_list.append(ground_main_lang_feat)
            ground_main_lang_len_list.append(ground_main_lang_len)
            ground_first_obj_list.append(ground_first_obj)

            # attr_masks_list.append(attr_masks)
            # attr_masks_num_list.append(attr_masks_num)
            # rel_masks_list.append(rel_masks)
            # rel_masks_num_list.append(rel_masks_num)
            all_masks_list.append(all_masks)
            all_masks_num_list.append(all_masks_num)

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
        
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]
        
        # ------------------------------- LABELS ------------------------------    
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))

        ref_box_label_list = []
        ref_center_label_list = []
        ref_heading_class_label_list = []
        ref_heading_residual_label_list = []
        ref_size_class_label_list = []
        ref_size_residual_label_list = []
        ref_box_corner_label_list = []

        ref_box_label = np.zeros(MAX_NUM_OBJ) # bbox label for reference target
        ref_center_label = np.zeros(3) # bbox center for reference target
        ref_heading_class_label = 0
        ref_heading_residual_label = 0
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3) # bbox size residual for reference target
        ref_box_corner_label = np.zeros((8, 3))

        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        
        num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
        target_bboxes_mask[0:num_bbox] = 1
        target_bboxes[0:num_bbox,:] = instance_bboxes[:MAX_NUM_OBJ,0:6]
        
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        if self.augment:
            if np.random.random() > 0.7:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                target_bboxes[:,0] = -1 * target_bboxes[:,0]                
                
            if np.random.random() > 0.7:
                # Flipping along the XZ plane
                point_cloud[:,1] = -1 * point_cloud[:,1]
                target_bboxes[:,1] = -1 * target_bboxes[:,1]                                

            # Rotation along X-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = rotx(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

            # Rotation along Y-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = roty(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")

            # print('Warning! Dont Use Extra Augmentation!(votenet didnot use it)', flush=True)
            # NEW: scale from 0.8 to 1.2
            # print(rot_mat.shape, point_cloud.shape, flush=True)
            scale = np.random.uniform(-0.1, 0.1, (3, 3))
            scale = np.exp(scale)
            # print(scale, '<<< scale', flush=True)
            scale = scale * np.eye(3)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], scale)
            if self.use_height:
                point_cloud[:, 3] = point_cloud[:, 3] * float(scale[2, 2])
            target_bboxes[:, 0:3] = np.dot(target_bboxes[:, 0:3], scale)
            target_bboxes[:, 3:6] = np.dot(target_bboxes[:, 3:6], scale)

            # Translation
            point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered 
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label. 
        for i_instance in np.unique(instance_labels):            
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label            
            if semantic_labels[ind[0]] in DC.nyu40ids:
                x = point_cloud[ind,:3]
                center = 0.5*(x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
        
        class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:num_bbox] = class_ind
        size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]

        # construct the reference target label for each bbox
        for j in range(self.lang_num_max):
            ref_box_label = np.zeros(MAX_NUM_OBJ)
            for i, gt_id in enumerate(instance_bboxes[:num_bbox,-1]):
                if gt_id == object_id_list[j]:
                    ref_box_label[i] = 1
                    ref_center_label = target_bboxes[i, 0:3]
                    ref_heading_class_label = angle_classes[i]
                    ref_heading_residual_label = angle_residuals[i]
                    ref_size_class_label = size_classes[i]
                    ref_size_residual_label = size_residuals[i]

                    # construct ground truth box corner coordinates
                    ref_obb = DC.param2obb(ref_center_label, ref_heading_class_label, ref_heading_residual_label,
                                    ref_size_class_label, ref_size_residual_label)
                    ref_box_corner_label = get_3d_box(ref_obb[3:6], ref_obb[6], ref_obb[0:3])

                    ref_box_label_list.append(ref_box_label)
                    ref_center_label_list.append(ref_center_label)
                    ref_heading_class_label_list.append(ref_heading_class_label)
                    ref_heading_residual_label_list.append(ref_heading_residual_label)
                    ref_size_class_label_list.append(ref_size_class_label)
                    ref_size_residual_label_list.append(ref_size_residual_label)
                    ref_box_corner_label_list.append(ref_box_corner_label)
        ref_num = len(ref_box_label_list)
        for j in range(self.lang_num_max-ref_num):
            ref_box_label_list.append(ref_box_label)
            ref_center_label_list.append(ref_center_label)
            ref_heading_class_label_list.append(ref_heading_class_label)
            ref_heading_residual_label_list.append(ref_heading_residual_label)
            ref_size_class_label_list.append(ref_size_class_label)
            ref_size_residual_label_list.append(ref_size_residual_label)
            ref_box_corner_label_list.append(ref_box_corner_label)
        # construct all GT bbox corners
        all_obb = DC.param2obb_batch(target_bboxes[:num_bbox, 0:3], angle_classes[:num_bbox].astype(np.int64), angle_residuals[:num_bbox],
                                size_classes[:num_bbox].astype(np.int64), size_residuals[:num_bbox])
        all_box_corner_label = get_3d_box_batch(all_obb[:, 3:6], all_obb[:, 6], all_obb[:, 0:3])

        # store
        gt_box_corner_label = np.zeros((MAX_NUM_OBJ, 8, 3))
        gt_box_masks = np.zeros((MAX_NUM_OBJ,))
        gt_box_object_ids = np.zeros((MAX_NUM_OBJ,))

        gt_box_corner_label[:num_bbox] = all_box_corner_label
        gt_box_masks[:num_bbox] = 1
        gt_box_object_ids[:num_bbox] = instance_bboxes[:, -1]

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_object_ids = np.zeros((MAX_NUM_OBJ,)) # object ids of all objects
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
            target_object_ids[0:num_bbox] = instance_bboxes[:, -1][0:num_bbox]
        except KeyError:
            pass

        object_cat_list = []
        for i in range(self.lang_num_max):
            object_cat = 17
            if object_name_list[i] in self.raw2label:
                object_cat = self.raw2label[object_name_list[i]]
            else:
                for tmp in object_name_list[i].split(" "):
                    if tmp in self.raw2label:
                        object_cat = self.raw2label[tmp]
            object_cat_list.append(object_cat)

        istrain = 0
        if self.split == "train":
            istrain = 1

        # object rotations
        scene_object_rotations = np.zeros((MAX_NUM_OBJ, 3, 3))
        scene_object_rotation_masks = np.zeros((MAX_NUM_OBJ,)) # NOTE this is not object mask!!!
        # if scene is not in scan2cad annotations, skip
        # if the instance is not in scan2cad annotations, skip
        if self.scan2cad_rotation and scene_id in self.scan2cad_rotation:
            for i, instance_id in enumerate(instance_bboxes[:num_bbox,-1].astype(int)):
                try:
                    rotation = np.array(self.scan2cad_rotation[scene_id][str(instance_id)])

                    scene_object_rotations[i] = rotation
                    scene_object_rotation_masks[i] = 1
                except KeyError:
                    pass

        data_dict = {}
        data_dict["unk"] = unk.astype(np.float32)
        data_dict["istrain"] = istrain
        data_dict["lang_num"] = np.array(lang_num).astype(np.int64)
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features
        data_dict["lang_feat"] = lang_feat.astype(np.float32) # language feature vectors
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64) # length of each description
        data_dict["lang_ids"] = np.array(self.lang_ids[scene_id][str(object_id)][ann_id]).astype(np.int64)

        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (MAX_NUM_OBJ,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (MAX_NUM_OBJ, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (MAX_NUM_OBJ,) semantic class index

        data_dict["scene_object_ids"] = target_object_ids.astype(np.int64) # (MAX_NUM_OBJ,) object ids of all objects
        data_dict["scene_object_rotations"] = scene_object_rotations.astype(np.float32) # (MAX_NUM_OBJ, 3, 3)
        data_dict["scene_object_rotation_masks"] = scene_object_rotation_masks.astype(np.int64) # (MAX_NUM_OBJ)
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict["dataset_idx"] = np.array(idx).astype(np.int64)
        data_dict["scan_idx"] = np.array(idx).astype(np.int64)
        data_dict["pcl_color"] = pcl_color

        data_dict["ref_box_label"] = ref_box_label.astype(np.int64) # 0/1 reference labels for each object bbox
        data_dict["ref_center_label"] = ref_center_label.astype(np.float32)
        data_dict["ref_heading_class_label"] = np.array(int(ref_heading_class_label)).astype(np.int64)
        data_dict["ref_heading_residual_label"] = np.array(int(ref_heading_residual_label)).astype(np.int64)
        data_dict["ref_size_class_label"] = np.array(int(ref_size_class_label)).astype(np.int64)
        data_dict["ref_size_residual_label"] = ref_size_residual_label.astype(np.float32)
        data_dict["ref_box_corner_label"] = ref_box_corner_label.astype(np.float64) # target box corners NOTE type must be double
        data_dict["gt_box_corner_label"] = gt_box_corner_label.astype(np.float64) # all GT box corners NOTE type must be double
        data_dict["gt_box_masks"] = gt_box_masks.astype(np.int64) # valid bbox masks
        data_dict["gt_box_object_ids"] = gt_box_object_ids.astype(np.int64) # valid bbox object ids
        data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)
        data_dict["ann_id"] = np.array(int(ann_id)).astype(np.int64)
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64)
        data_dict["unique_multiple"] = np.array(self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]).astype(np.int64)
        data_dict["pcl_color"] = pcl_color
        data_dict["load_time"] = time.time() - start

        data_dict["lang_feat_list"] = np.array(lang_feat_list).astype(np.float32)  # language feature vectors
        data_dict["lang_len_list"] = np.array(lang_len_list).astype(np.int64)  # length of each description
        data_dict["lang_ids_list"] = np.array(lang_ids_list).astype(np.int64)
        data_dict["main_lang_feat_list"] = np.array(main_lang_feat_list).astype(np.float32)  # main language feature vectors
        data_dict["main_lang_len_list"] = np.array(main_lang_len_list).astype(np.int64)  # length of each main description
        data_dict["first_obj_list"] = np.array(first_obj_list).astype(np.int64)
        data_dict["unk_list"] = np.array(unk_list).astype(np.float32)

        data_dict["ground_lang_feat_list"] = np.array(ground_lang_feat_list).astype(np.float32)  # language feature vectors
        data_dict["ground_lang_len_list"] = np.array(ground_lang_len_list).astype(np.int64)  # length of each description
        data_dict["ground_lang_ids_list"] = np.array(ground_lang_ids_list).astype(np.int64)
        data_dict["ground_main_lang_feat_list"] = np.array(ground_main_lang_feat_list).astype(np.float32)  # main language feature vectors
        data_dict["ground_main_lang_len_list"] = np.array(ground_main_lang_len_list).astype(np.int64)  # length of each main description
        data_dict["ground_first_obj_list"] = np.array(ground_first_obj_list).astype(np.int64)

        # data_dict['attr_masks_list'] = np.array(attr_masks_list).astype(np.int64)
        # data_dict['attr_masks_num_list'] = np.array(attr_masks_num_list).astype(np.int64)
        # data_dict['rel_masks_list'] = np.array(rel_masks_list).astype(np.int64)
        # data_dict['rel_masks_num_list'] = np.array(rel_masks_num_list).astype(np.int64)
        data_dict['all_masks_list'] = np.array(all_masks_list).astype(np.int64)
        data_dict['all_masks_num_list'] = np.array(all_masks_num_list).astype(np.int64)

        data_dict["ref_box_label_list"] = np.array(ref_box_label_list).astype(np.int64)  # 0/1 reference labels for each object bbox
        data_dict["ref_center_label_list"] = np.array(ref_center_label_list).astype(np.float32)
        data_dict["ref_heading_class_label_list"] = np.array(ref_heading_class_label_list).astype(np.int64)
        data_dict["ref_heading_residual_label_list"] = np.array(ref_heading_residual_label_list).astype(np.int64)
        data_dict["ref_size_class_label_list"] = np.array(ref_size_class_label_list).astype(np.int64)
        data_dict["ref_size_residual_label_list"] = np.array(ref_size_residual_label_list).astype(np.float32)
        data_dict["ref_box_corner_label_list"] = np.array(ref_box_corner_label_list).astype(np.float64)
        data_dict["object_id_list"] = np.array(object_id_list).astype(np.int64)
        data_dict["ann_id_list"] = np.array(ann_id_list).astype(np.int64)
        data_dict["easy_hard_list"] = np.array(easy_hard_list).astype(np.int64)
        data_dict["dep_indep_list"] = np.array(dep_indep_list).astype(np.int64)
        data_dict["object_cat_list"] = np.array(object_cat_list).astype(np.int64)

        # data_dict["idx"] = idx
        # data_dict["scene_id"] = scene_id

        if not CONF.pretrain_model_on:
            for key in CONF.PRETRAINED_LIST:
                data_dict[key] = np.array(self.pretrained_data[scene_id][key]).astype(np.float32)

        unique_multiple_list = []
        for i in range(self.lang_num_max):
            object_id = object_id_list[i]
            ann_id = ann_id_list[i]
            unique_multiple = self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]
            unique_multiple_list.append(unique_multiple)
        data_dict["unique_multiple_list"] = np.array(unique_multiple_list).astype(np.int64)

        return data_dict


class ScannetReferenceTestDataset():
       
    def __init__(self, scanrefer_all_scene, 
        num_points=40000,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False):

        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview

        # load data
        self.scene_data = self._load_data()
        self.glove = pickle.load(open(GLOVE_PICKLE, "rb"))
        self.vocabulary = json.load(open(SCANREFER_VOCAB))
        self.multiview_data = {}
       
    def __len__(self):
        return len(self.scanrefer_all_scene)

    def __getitem__(self, idx):
        start = time.time()

        scene_id = self.scanrefer_all_scene[idx]

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
        
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        

        data_dict = {}
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features
        data_dict["dataset_idx"] = idx
        data_dict["lang_feat"] = self.glove["sos"].astype(np.float32) # GloVE embedding for sos token
        data_dict["load_time"] = time.time() - start

        return data_dict

    def _load_data(self):
        scene_data = {}
        for scene_id in self.scanrefer_all_scene:
            scene_data[scene_id] = {}
            scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned

        return scene_data
