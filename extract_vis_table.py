import os
import sys
import json
import csv
import numpy as np

SCANNET_V2_TSV = "/data/huanghaifeng/3DWSVG/data/scannet/meta_data/scannetv2-labels.combined.tsv"
type2class = {'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
              'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
              'refrigerator': 12, 'shower curtain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16, 'others': 17}

raw2label = {}

def get_raw2label():
    # mapping
    scannet_labels = type2class.keys()
    scannet2label = {label: i for i, label in enumerate(scannet_labels)}

    lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
    lines = lines[1:]
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


output_dir = "/data/huanghaifeng/3DWSVG/outputs/test"
model_names = ["ALL_SCAN_8_2022-10-29_19-45-13", "MIL_SCAN_2022-11-01_11-39-07", "MARGIN_SCAN_2022-11-03_00-39-55"]
folders = [os.path.join(output_dir, name, "vis") for name in model_names]
scannet_val = json.load(open("/data/huanghaifeng/3DWSVG/data/ScanRefer_filtered_val.json"))

data = {}
all_sem_labels = {}
get_raw2label()
for item in scannet_val:
    scene_id = item["scene_id"]
    object_id = item["object_id"]
    object_name = " ".join(item["object_name"].split("_"))
    ann_id = item["ann_id"]
    ann = item["description"]
    if scene_id not in data:
        data[scene_id] = {}
        all_sem_labels[scene_id] = []
    if object_id not in data[scene_id]:
        data[scene_id][object_id] = {}
        try:
            all_sem_labels[scene_id].append(raw2label[object_name])
        except KeyError:
            all_sem_labels[scene_id].append(17)
    if ann_id not in data[scene_id][object_id]:
        data[scene_id][object_id][ann_id] = {}
        data[scene_id][object_id][ann_id]["ious"] = []
        data[scene_id][object_id][ann_id]["files"] = []
        data[scene_id][object_id][ann_id]["object_name"] = object_name
        data[scene_id][object_id][ann_id]["ann"] = ann

all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

for folder in folders:
    for scene_id in os.listdir(folder):
        if "gt_files" not in data[scene_id]:
            data[scene_id]["gt_files"] = []
            for file in os.listdir(os.path.join(folder, scene_id)):
                if file[:2] == "gt":
                    data[scene_id]["gt_files"].append(file)
        for file in os.listdir(os.path.join(folder, scene_id)):
            if file[:4] == "pred":
                tmp = file.split("_")
                object_id = tmp[1]
                object_name = tmp[2] if len(tmp) == 6 else "_".join(tmp[2:4])
                ann_id = tmp[-3]
                iou = float(tmp[-1][:-4])
                # print(scene_id, object_id, ann_id, object_name, iou)
                data[scene_id][object_id][ann_id]["ious"].append(iou)
                data[scene_id][object_id][ann_id]["files"].append(file)
                # if "gt_file" not in data[scene_id][object_id][ann_id]:
                #     gt_file = "gt_{}_{}.ply".format(object_id, object_name)
                #     data[scene_id][object_id][ann_id]["gt_file"] = gt_file

labels = ["scene_id", "object_id", "ann_id", "object_name", "annotation", "same_count"]
for name in model_names:
    labels.append(name)
for i in range(len(model_names)):
    labels.append("file" + str(i+1))
labels.append("gt_file")

outputs = [labels, ]

for scene_id, scene_data in data.items():
    for object_id, object_data in scene_data.items():
        if object_id != "gt_files":
            for ann_id, ann_data in object_data.items():
                object_name = ann_data["object_name"]
                item = [scene_id, object_id, ann_id, object_name, ann_data["ann"]]
                ious = ann_data["ious"]
                flag = 1
                for i in range(len(ious) - 1):
                    if ious[i] < ious[i + 1]:
                        flag = 0
                    # if ious[i+1] > 0 and ious[i] / ious[i+1] < 1.2:
                    #     flag = 0
                if not flag:
                    continue
                try:
                    sem_label = raw2label[object_name]
                except KeyError:
                    sem_label = 17
                # unique = 1 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 0
                unique = (all_sem_labels[scene_id] == sem_label).sum()
                item.append(unique)
                item.extend(ann_data["ious"])
                item.extend(ann_data["files"])
                # item.append(ann_data["gt_file"])
                item.append(scene_data["gt_files"])
                outputs.append(item)
print(len(outputs) - 1)

output_dir = "vis_table.csv"
with open(output_dir, "w") as f:
    csvwriter = csv.writer(f)
    csvwriter.writerows(outputs)
