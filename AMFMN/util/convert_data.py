# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Convert image features from bottom up attention to numpy array"""
import os
import base64
import csv
import sys
import zlib
import json
import argparse

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--imgid_list', default='../data/coco_precomp/train_ids.txt',
                    help='Path to list of image id')
parser.add_argument('--input_file', default=['../data/bu_data/trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0'],
                                             # ,'../data/bu_data/trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1'],
                    help='tsv of all image data (output of bottom-up-attention/tools/generate_tsv.py), \
                    where each columns are: [image_id, image_w, image_h, num_boxes, boxes, features].')
parser.add_argument('--output_dir', default='../data/coco_precomp/',
                    help='Output directory.')
parser.add_argument('--split', default='train',
                    help='train|dev|test')
opt = parser.parse_args()
print(opt)


meta = []
feature = {}
for line in open(opt.imgid_list):
    sid = int(line.strip())
    meta.append(sid)
    feature[sid] = None

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

if __name__ == '__main__':
    for input_file in opt.input_file:
        with open(input_file, "r+t") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
            for item in reader:
                item['image_id'] = int(item['image_id'])
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])
                item['num_boxes'] = int(item['num_boxes'])
                for field in ['boxes', 'features']:
                    item[field] = np.frombuffer(base64.decodestring(item[field].encode()),
                            dtype=np.float32).reshape((item['num_boxes'],-1))
                if item['image_id'] in feature:
                    feature[item['image_id']] = item['features']

    # Padding
    data_out = []
    for sid in meta:
        padding_data = np.zeros((36,2048))
        region_num = np.shape(feature[sid])[0]
        if region_num <= 36:
            padding_data[:region_num, :] = feature[sid]
        else:
            padding_data = feature[sid][:36, :]
        data_out.append(padding_data)

    data_out = np.stack(data_out, axis=0)
    print("Final numpy array shape:", data_out.shape)
    np.save(os.path.join(opt.output_dir, '{}_ims.npy'.format(opt.split)), data_out)