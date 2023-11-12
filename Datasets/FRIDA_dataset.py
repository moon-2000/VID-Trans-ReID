from __future__ import print_function, absolute_import
import os
import json
from collections import defaultdict
import random

class FRIDA(object):
    """
    FRIDA Dataset
    Args:
        data_dir (str): Path to the root directory of FRIDA dataset.
        min_seq_len (int): Tracklet with length shorter than this value will be discarded (default: 0).
    """
    def __init__(self, data_dir = '/content/FRIDA', min_seq_len=0):
        self.data_dir = data_dir
        self.train_dirs = [f"Segment_{i + 1}" for i in range(4)]  # FRIDA has 4 segments
        self.cameras = ['Camera_1', 'Camera_2', 'Camera_3']  # FRIDA has 3 cameras
        self._check_before_run()

        self.train, self.test, num_train_tracklets, num_test_tracklets, num_train_pids, num_test_pids, num_imgs_train, num_imgs_test = \
            self._process_data(self.train_dirs, split_ratio=0.5, num_train_ids=10)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_test
        min_num = min(num_imgs_per_tracklet)
        max_num = max(num_imgs_per_tracklet)
        avg_num = sum(num_imgs_per_tracklet) / (num_train_tracklets + num_test_tracklets)

        num_total_pids = num_train_pids + num_test_pids
        num_total_tracklets = num_train_tracklets + num_test_tracklets

        print("=> FRIDA loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  test     | {:5d} | {:8d}".format(num_test_pids, num_test_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = self._create_query_gallery(self.train)
        self.test = self._create_query_gallery(self.test)

        self.num_train_pids = num_train_pids
        self.num_test_pids = num_test_pids
        self.num_train_cams = len(self.cameras)
        self.num_test_cams = len(self.cameras)
        self.num_train_vids = num_train_tracklets
        self.num_test_vids = num_test_tracklets

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))

    def _process_data(self, dirnames, split_ratio=0.5, num_train_ids=10):
        tracklets_train = []
        tracklets_test = []
        num_imgs_per_tracklet_train = []
        num_imgs_per_tracklet_test = []
        pid_container = set()

        for segment in dirnames:
            for camera in self.cameras:
                json_file = os.path.join(self.data_dir, 'Annotations', segment, camera, 'data2.json')
                with open(json_file, 'r') as f:
                    data = json.load(f)

                selected_persons = random.sample(data, num_train_ids)
                
                for person_info in data:
                    img_id = person_info['file_name']
                    pid = person_info['person_id']
                    person_id = f'person_{str(pid).zfill(2)}'  # Convert integer ID to zero-padded string
                    pid_container.add(person_id)  # Use the zero-padded ID for consistency
                    img_path = os.path.join(self.data_dir, 'BBs', segment, person_id, camera, img_id)
                    
                    if person_info in selected_persons:
                        tracklets_train.append((img_path, person_id, self.cameras.index(camera)))
                        num_imgs_per_tracklet_train.append(1)
                    else:
                        tracklets_test.append((img_path, person_id, self.cameras.index(camera)))
                        num_imgs_per_tracklet_test.append(1)

        num_train_tracklets = len(tracklets_train)
        num_test_tracklets = len(tracklets_test)
        num_train_pids = len(pid_container)
        num_test_pids = len(pid_container) - num_train_pids

        return tracklets_train, tracklets_test, num_train_tracklets, num_test_tracklets, num_train_pids, num_test_pids, num_imgs_per_tracklet_train, num_imgs_per_tracklet_test

    def _create_query_gallery(self, tracklets):
        gallery = defaultdict(dict)
        query = defaultdict(dict)

        for tracklet in tracklets:
            img_path, person_id, camera_idx = tracklet
            if camera_idx == 0:  # Camera A
                query[person_id][camera_idx] = img_path
            else:  # Cameras B and C
                gallery[person_id][camera_idx] = img_path

        return {'query': query, 'gallery': gallery}
