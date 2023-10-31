from __future__ import print_function, absolute_import
from collections import defaultdict
import json
import glob
import os.path as osp
import numpy as np

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

class FRIDA(object):
    root = "FRIDA"  # Update this path to the root directory of the FRIDA dataset
    annotations_path = osp.join(root, "annotations.json")  # Path to FRIDA annotations file

    def __init__(self, min_seq_len=0):
        self._check_before_run()
        annotations = read_json(self.annotations_path)

        tracklets, num_tracklets, num_pids, num_imgs_per_tracklet = self._process_data(annotations)
        
        num_imgs_per_tracklet = np.array(num_imgs_per_tracklet)
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_pids
        num_total_tracklets = num_tracklets

        print("=> FRIDA loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_pids, num_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = tracklets
        self.num_train_pids = num_pids
        self.num_train_cams = 3
        self.num_train_vids = num_tracklets

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.annotations_path):
            raise RuntimeError("'{}' is not available".format(self.annotations_path))

    def _process_data(self, annotations):
        tracklets = []
        num_imgs_per_tracklet = []
        pid_set = set()

        for segment_id, segment_info in annotations.items():
            for camera_id, camera_info in segment_info.items():
                for person_id, frames in camera_info.items():
                    img_names = [osp.join(self.root, f"{segment_id}/camera{camera_id}/{frame}") for frame in frames]
                    pid_set.add(person_id)
                    tracklets.append((img_names, person_id, camera_id - 1))  # Camera IDs in code start from 0
                    num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(pid_set)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
