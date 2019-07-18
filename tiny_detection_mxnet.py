from __future__ import absolute_import
from __future__ import division

from tiny_faces.tiny_fd import TinyFacesDetector
import sys
import cv2
import os
import os.path as ops
import numpy as np
import json

import argparse


from manage_data.get_density_map import interpolate_scale
from manage_data.utils import mkdir_if_missing

parser = argparse.ArgumentParser(description='Tiny face detection for crowd counting')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='ucf-cc-50', help="dataset to create density masks")
parser.add_argument('-r', '--root-dir', type=str, default='/workspace/quispe/', help="root directory for datasets")
parser.add_argument('--save-plots', action='store_true', help="save plots of detected and interpolated tiny faces")
args = parser.parse_args()

detector = TinyFacesDetector(model_root='./tiny_faces/', prob_thresh=0.5, gpu_idx=0)

def UCF_CC_50():
    base_dir = ops.join(args.root_dir, "ucf_cc_50/UCF_CC_50")
    print("working over '{}'".format(base_dir))
    img_dir_path = os.path.join(base_dir, "images")
    den_dir_path = os.path.join(base_dir, "density_maps")
    est_dir_path = os.path.join(base_dir, "faces")
    lab_dir_path = os.path.join(base_dir, "labels")
    return img_dir_path, den_dir_path, est_dir_path, lab_dir_path

def shanghai_tech(part, mode):
    base_dir = ops.join(args.root_dir, "ShanghaiTech/" + part + "/" + mode)
    print("working over '{}'".format(base_dir))
    img_dir_path = os.path.join(base_dir, "images")
    den_dir_path = os.path.join(base_dir, "density_maps")
    est_dir_path = os.path.join(base_dir, "faces")
    lab_dir_path = os.path.join(base_dir, "labels")
    return img_dir_path, den_dir_path, est_dir_path, lab_dir_path

def run_face_detector(img_dir_path, den_dir_path, est_dir_path, lab_dir_path):
    mkdir_if_missing(est_dir_path)
    file_names = os.listdir(img_dir_path)
    file_names.sort()
    mae, mse = 0, 0
    for file_name in file_names:
        file_extention = file_name.split('.')[-1]
        file_id = file_name[:len(file_name) - len(file_extention)]

        file_path = os.path.join(img_dir_path, file_name)
        density_map_path = os.path.join(den_dir_path, file_id + 'npy')
        label_path = os.path.join(lab_dir_path, file_id + 'json')

        img = cv2.imread(file_path)
        boxes = detector.detect(img)
        save_path = os.path.join(est_dir_path, file_id + 'npy')
        np.save(save_path, boxes)

        gt = np.load(density_map_path)
        gt_count = np.sum(gt)
        et_count = boxes.shape[0]
        mae += abs(gt_count-et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))

        if args.save_plots:
            for r in boxes:
                cv2.rectangle(img, (r[0],r[1]), (r[2],r[3]), (255,255,0), 1)
            points = [[p['y'], p['x']] for p in json.load(open(label_path))]
            _, bb_sizes = interpolate_scale(img.shape, points, boxes)
            for p, r in zip(points, bb_sizes):
                cv2.rectangle(img, (r[0],r[1]), (r[2],r[3]), (255,0, 255), 1)
                cv2.circle(img, (int(p[1]), int(p[0])), 1, (255,0, 255), 2)

            #img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            print('Image {}: gt count {:.1f}, est count {}'.format(file_id, gt_count, et_count))
            if not os.path.exists(est_dir_path):
                os.mkdir(est_dir_path)
            save_path = os.path.join(est_dir_path, file_id + 'jpg')
            print (save_path)
            cv2.imwrite(save_path, img)

    mae = mae/len(file_names)
    mse = np.sqrt(mse/len(file_names))
    print("final mse: {:.3f}, final mae: {:.3f}".format(mse, mae))

if __name__ == '__main__':
    if args.dataset == 'ucf-cc-50':
        img_dir_path, den_dir_path, est_dir_path, lab_dir_path = UCF_CC_50()
        run_face_detector(img_dir_path, den_dir_path, est_dir_path, lab_dir_path)
    if args.dataset == 'shanghai-tech':
        img_dir_path, den_dir_path, est_dir_path, lab_dir_path = shanghai_tech("part_A", "train_data")
        run_face_detector(img_dir_path, den_dir_path, est_dir_path, lab_dir_path)
        img_dir_path, den_dir_path, est_dir_path, lab_dir_path = shanghai_tech("part_A", "test_data")
        run_face_detector(img_dir_path, den_dir_path, est_dir_path, lab_dir_path)
        img_dir_path, den_dir_path, est_dir_path, lab_dir_path = shanghai_tech("part_B", "train_data")
        run_face_detector(img_dir_path, den_dir_path, est_dir_path, lab_dir_path)
        img_dir_path, den_dir_path, est_dir_path, lab_dir_path = shanghai_tech("part_B", "test_data")
        run_face_detector(img_dir_path, den_dir_path, est_dir_path, lab_dir_path)
