import functools

import nltk
import torch
import json
import os
import csv
import sys
import ast
from nltk.parse.stanford import StanfordParser
from nltk.tree import *
# stanford_parser_dir = r'/home/huanghaifeng/stanford-parser-full-2015-12-09'
# eng_model_path = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
# path_to_models_jar = stanford_parser_dir + "/stanford-parser-3.6.0-models.jar"
# path_to_jar = stanford_parser_dir + "/stanford-parser.jar"
# PATH_DATA = '/home/huanghaifeng/3DWSVG/data'
PATH_DATA = '/root/3DWSVG/data'
# SCANREFER_TRAIN = json.load(open(os.path.join(PATH_DATA, "ScanRefer_filtered_train.json")))
# SCANREFER_VAL = json.load(open(os.path.join(PATH_DATA, "ScanRefer_filtered_val.json")))[0:10]

SR3D_TRAIN = json.load(open(os.path.join(PATH_DATA, "sr3d", "masked_sr3d_train.json")))
SR3D_VAL = json.load(open(os.path.join(PATH_DATA, "sr3d", "masked_sr3d_test.json")))
SR3D_IDS = None

sr3d_new = {}
for data in SR3D_TRAIN:
    sr3d_new[data["ann_id"]] = data
for data in SR3D_VAL:
    sr3d_new[data["ann_id"]] = data

SR3D_DATA = []
with open(os.path.join(PATH_DATA, "sr3d/sr3d.csv")) as f:
    csvreader = csv.reader(f)
    SR3D_IDS = csvreader.__next__()
    for row in csvreader:
        SR3D_DATA.append(row)

bad_ids = []
with open(os.path.join(PATH_DATA, "sr3d/manually_inspected_bad_contexts.csv")) as f:
    csvreader = csv.reader(f)
    csvreader.__next__()
    for row in csvreader:
        bad_ids.append(row[0])
# print(bad_ids)

train_set = []
test_set = []
with open(os.path.join(PATH_DATA, "sr3d/train_scans.txt")) as f:
    train_set = f.read()
with open(os.path.join(PATH_DATA, "sr3d/test_scans.txt")) as f:
    test_set = f.read()

print("tot:", len(train_set) + len(test_set))

id2idx = {}
for i in range(len(SR3D_IDS)):
    id2idx[SR3D_IDS[i]] = i
print(id2idx)
print(len(SR3D_DATA))

train_data = []
test_data = []
# print(sr3d_new.keys())
for i, data in enumerate(SR3D_DATA):
    print(i)
    if data[id2idx["stimulus_id"]] in bad_ids:
        continue
    tmp = sr3d_new[i]
    tmp["hardness"] = data[id2idx["stimulus_id"]].split('-', maxsplit=4)[2]
    # tmp["scene_id"] = data[id2idx["scan_id"]]
    # tmp["object_id"] = data[id2idx["target_id"]]
    # tmp["object_name"] = data[id2idx["instance_type"]]
    # tmp["ann_id"] = i
    # tmp["description"] = data[id2idx["utterance"]]
    # tmp["token"] = ast.literal_eval(data[id2idx["tokens"]])
    # tmp["masked_sentence"] = data[id2idx["masked_sentence"]]
    # tmp["mask_list"] = ast.literal_eval(data[id2idx["mask_list"]])
    if tmp["scene_id"] in test_set:
        test_data.append(tmp)
    else:
        train_data.append(tmp)
    # print(tmp)
    # sys.exit()


def my_compare(x, y):
    if x["scene_id"] > y["scene_id"]:
        return 1
    elif x["scene_id"] < y["scene_id"]:
        return -1
    return 0


train_data.sort(key=functools.cmp_to_key(my_compare))
test_data.sort(key=functools.cmp_to_key(my_compare))


with open(os.path.join(PATH_DATA, "sr3d/masked_sr3d_train.json"), "w") as f:
    json.dump(train_data, f)

with open(os.path.join(PATH_DATA, "sr3d/masked_sr3d_test.json"), "w") as f:
    json.dump(test_data, f)

# NR3D_DATA = []
# NR3D_IDS = None
#
# with open(os.path.join(PATH_DATA, "nr3d/masked_nr3d.csv")) as f:
#     csvreader = csv.reader(f)
#     NR3D_IDS = csvreader.__next__()
#     for row in csvreader:
#         NR3D_DATA.append(row)
#
# bad_ids = []
# with open(os.path.join(PATH_DATA, "nr3d/manually_inspected_bad_contexts.csv")) as f:
#     csvreader = csv.reader(f)
#     csvreader.__next__()
#     for row in csvreader:
#         bad_ids.append(row[0])
# # print(bad_ids)
#
# train_set = []
# test_set = []
# with open(os.path.join(PATH_DATA, "nr3d/train_scans.txt")) as f:
#     train_set = f.read()
# with open(os.path.join(PATH_DATA, "nr3d/test_scans.txt")) as f:
#     test_set = f.read()
#
# id2idx = {}
# for i in range(len(NR3D_IDS)):
#     id2idx[NR3D_IDS[i]] = i
# print(id2idx)
# # print(NR3D_IDS)
# print(len(NR3D_DATA))
# # sys.exit()
#
# train_data = []
# test_data = []
# for i, data in enumerate(NR3D_DATA):
#     print(i)
#     if data[id2idx["stimulus_id"]] in bad_ids:
#         continue
#     tmp = {}
#     tmp["scene_id"] = data[id2idx["scan_id"]]
#     tmp["object_id"] = data[id2idx["target_id"]]
#     tmp["object_name"] = data[id2idx["instance_type"]]
#     tmp["ann_id"] = data[id2idx["assignmentid"]]
#     tmp["description"] = data[id2idx["utterance"]]
#     tmp["token"] = ast.literal_eval(data[id2idx["tokens"]])
#     tmp["masked_sentence"] = data[id2idx["masked_sentence"]]
#     tmp["mask_list"] = ast.literal_eval(data[id2idx["mask_list"]])
#     tmp["hardness"] = data[id2idx["stimulus_id"]].split('-', maxsplit=4)[2]
#     print(tmp["hardness"])
#     if tmp["scene_id"] in train_set:
#         train_data.append(tmp)
#     else:
#         test_data.append(tmp)
#     # print(tmp)
#     # sys.exit()
#
#
# def my_compare(x, y):
#     if x["scene_id"] > y["scene_id"]:
#         return 1
#     elif x["scene_id"] < y["scene_id"]:
#         return -1
#     return 0
#
#
# train_data.sort(key=functools.cmp_to_key(my_compare))
# test_data.sort(key=functools.cmp_to_key(my_compare))
#
#
# with open(os.path.join(PATH_DATA, "nr3d/masked_nr3d_train.json"), "w") as f:
#     json.dump(train_data, f)
#
# with open(os.path.join(PATH_DATA, "nr3d/masked_nr3d_test.json"), "w") as f:
#     json.dump(test_data, f)

sys.exit()

def MultiTreePaths(root):
    def helper(root, path, res):
        if type(root)==str:
            res.append(path +str(root))
            return
        l = len(root)
        for i in range(l):
                if len(root)>=i:
                    if root[i]:
                        helper(root[i], path +str(root.label()) +'->', res)
    if root is None:
        return []
    l = []
    helper(root, '', l)
    return l


def find_JJ(tokens, leaf_nodes):
    masked_tokens = tokens
    mask_list = []
    for i, word in enumerate(leaf_nodes):
        if 'JJ' in word:
            mask_list.append(1)
            masked_tokens[i] = '[MASK]'
        else:
            mask_list.append(0)
    return masked_tokens, mask_list


def find_PPNN(tokens, leaf_nodes):
    masked_tokens = tokens
    mask_list = []
    for i, word in enumerate(leaf_nodes):
        if 'PP' in word:
            if 'NN' in word:
                mask_list.append(1)
                masked_tokens[i] = '[MASK]'
            else:
                mask_list.append(0)
        else:
            mask_list.append(0)
    return masked_tokens, mask_list


parser = StanfordParser(model_path=eng_model_path, path_to_models_jar=path_to_models_jar, path_to_jar=path_to_jar)

# nr3d
# NR3D_IDS.extend(["masked_sentence", "mask_list"])
#
# for i, data in enumerate(NR3D_DATA):
#     sentence = data[2]
#     tokens = nltk.word_tokenize(sentence)
#
#     tree = list(parser.parse(tokens))[0]
#     leaf_nodes = MultiTreePaths(tree[0])
#     PPNN_masked_tokens, PPNN_mask_list = find_PPNN(tokens, leaf_nodes)
#     data.append(' '.join(PPNN_masked_tokens))  # masked_sentence
#     data.append(PPNN_mask_list)
#
#     print(i)
#     print(data[-2:])
#
# with open(os.path.join(PATH_DATA, "nr3d/masked_nr3d.csv"), "w") as f:
#     csvwriter = csv.writer(f)
#     csvwriter.writerow(NR3D_IDS)
#     csvwriter.writerows(NR3D_DATA)

# scanrefer
# NEW_SR3D_TRAIN = []
# for i, data in enumerate(SR3D_TRAIN):
#     sentence = data['description']
#     first_sentence = nltk.sent_tokenize(sentence)[0]
#     tokens = nltk.word_tokenize(sentence)
#     first_tokens = nltk.word_tokenize(first_sentence)
#
#     # first_tree = list(parser.parse(first_tokens))[0]
#     # first_leaf_nodes = MultiTreePaths(first_tree[0])
#     # JJ_masked_tokens, JJ_mask_list = find_JJ(first_tokens, first_leaf_nodes)
#     # data['masked_first_sentence'] = ' '.join(JJ_masked_tokens)
#     # data['mask_first_list'] = JJ_mask_list
#
#     tree = list(parser.parse(tokens))[0]
#     leaf_nodes = MultiTreePaths(tree[0])
#     PPNN_masked_tokens, PPNN_mask_list = find_PPNN(tokens, leaf_nodes)
#     data['masked_sentence'] = ' '.join(PPNN_masked_tokens)
#     data['mask_list'] = PPNN_mask_list
#     print("train", i)
#     NEW_SR3D_TRAIN.append(data)
#
# NEW_SR3D_VAL = []
# for i, data in enumerate(SR3D_VAL):
#     sentence = data['description']
#     first_sentence = nltk.sent_tokenize(sentence)[0]
#     tokens = nltk.word_tokenize(sentence)
#     first_tokens = nltk.word_tokenize(first_sentence)
#
#     # first_tree = list(parser.parse(first_tokens))[0]
#     # first_leaf_nodes = MultiTreePaths(first_tree[0])
#     # JJ_masked_tokens, JJ_mask_list = find_JJ(first_tokens, first_leaf_nodes)
#     # data['masked_first_sentence'] = ' '.join(JJ_masked_tokens)
#     # data['mask_first_list'] = JJ_mask_list
#
#     tree = list(parser.parse(tokens))[0]
#     leaf_nodes = MultiTreePaths(tree[0])
#     PPNN_masked_tokens, PPNN_mask_list = find_PPNN(tokens, leaf_nodes)
#     data['masked_sentence'] = ' '.join(PPNN_masked_tokens)
#     data['mask_list'] = PPNN_mask_list
#
#     print("val", i)
#     NEW_SR3D_VAL.append(data)
#
# with open(os.path.join(PATH_DATA, "masked_sr3d_train.json"), 'w') as f:
#     f.write(json.dumps(NEW_SR3D_TRAIN, indent=4, ensure_ascii=False))
#
# with open(os.path.join(PATH_DATA, "masked_sr3d_test.json"), 'w') as f:
#     f.write(json.dumps(NEW_SR3D_VAL, indent=4, ensure_ascii=False))

# scanrefer
# NEW_SCANREFER_TRAIN = []
# for data in SCANREFER_TRAIN:
#     sentence = data['description']
#     first_sentence = nltk.sent_tokenize(sentence)[0]
#     tokens = nltk.word_tokenize(sentence)
#     first_tokens = nltk.word_tokenize(first_sentence)
#
#     first_tree = list(parser.parse(first_tokens))[0]
#     first_leaf_nodes = MultiTreePaths(first_tree[0])
#     JJ_masked_tokens, JJ_mask_list = find_JJ(first_tokens, first_leaf_nodes)
#     data['masked_first_sentence'] = ' '.join(JJ_masked_tokens)
#     data['mask_first_list'] = JJ_mask_list
#
#     tree = list(parser.parse(tokens))[0]
#     leaf_nodes = MultiTreePaths(tree[0])
#     PPNN_masked_tokens, PPNN_mask_list = find_PPNN(tokens, leaf_nodes)
#     data['masked_sentence'] = ' '.join(PPNN_masked_tokens)
#     data['mask_list'] = PPNN_mask_list
#
#     NEW_SCANREFER_TRAIN.append(data)
#
#
# with open(os.path.join(PATH_DATA,'scanrefer/Masked_ScanRefer_filtered_train.json'), 'w') as f:
#     f.write(json.dumps(NEW_SCANREFER_TRAIN, indent=4, ensure_ascii=False))

