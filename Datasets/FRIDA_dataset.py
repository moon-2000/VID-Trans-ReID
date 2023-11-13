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
            self._process_data(self.train_dirs, min_seq_len=0, num_train_ids=10)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_test
        min_num = min(num_imgs_per_tracklet)
        max_num = max(num_imgs_per_tracklet)
        avg_num = sum(num_imgs_per_tracklet) / (num_train_tracklets + num_test_tracklets)

        num_total_pids = num_train_pids + num_test_pids
        num_total_tracklets = num_train_tracklets + num_test_tracklets

        

        self.num_train_pids = num_train_pids
        self.num_test_pids = num_test_pids
        self.num_train_cams = len(self.cameras)
        self.num_test_cams = len(self.cameras)
        self.num_train_vids = num_train_tracklets
        self.num_test_vids = num_test_tracklets

        # self.train = self._create_query_gallery(self.train)
        print(f"First 3 tracklets in train: {self.train[:3]}")

        query, gallery, tracklet_query, tracklet_gallery, num_query_pids, num_gallery_pids =  self._create_query_gallery(self.test)
        self.query = query
        self.gallery = gallery
        print(f"First 3 tracklets in query: {self.query[:3]}")
        
        num_query_tracklets = tracklet_query
        num_gallery_tracklets =  tracklet_gallery
        

        print("=> FRIDA loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(self.num_train_pids, self.num_train_vids))
        print("  test     | {:5d} | {:8d}".format(self.num_test_pids, self.num_test_vids))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))



    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))

    def _process_data(self, dirnames, min_seq_len=0, num_train_ids=10):
        tracklets_train = []
        tracklets_test = []
        num_imgs_per_tracklet_train = []
        num_imgs_per_tracklet_test = []
        pid_container = list(range(20))  # Assuming 20 persons in total

        # Randomly shuffle the list of person IDs
        random.shuffle(pid_container)

        # Select the first num_train_ids for training, and the rest for testing
        selected_persons_train = pid_container[:num_train_ids]
        selected_persons_test = pid_container[num_train_ids:]

        for segment in dirnames:
            for camera in self.cameras:
                json_file = os.path.join(self.data_dir, 'Annotations', segment, camera, 'data2.json')
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # for person_info in data:
                #     img_id = person_info['image_id']
                #     pid = person_info['person_id']
                #     person_id = f'person_{str(pid).zfill(2)}'  # Convert integer ID to zero-padded string
                #     img_path = os.path.join(self.data_dir, 'BBs', segment, img_id, camera, f'{person_id}.jpg')

                #     if pid in selected_persons_train:
                #         tracklets_train.append((img_path, person_id, self.cameras.index(camera)))
                #         num_imgs_per_tracklet_train.append(1)
                #     elif pid in selected_persons_test:
                #         tracklets_test.append((img_path, person_id, self.cameras.index(camera)))
                #         num_imgs_per_tracklet_test.append(1)
                for person_info in data:
                    img_id = person_info['image_id']
                    pid = person_info['person_id']
                    person_id = f'person_{str(pid).zfill(2)}'  # Convert integer ID to zero-padded string
                    img_path = os.path.join(self.data_dir, 'BBs', segment, img_id, camera, f'{person_id}.jpg')

                    tracklet = [(img_path, person_id, self.cameras.index(camera))]

                    if pid in selected_persons_train:
                        tracklets_train.append(tracklet)
                        num_imgs_per_tracklet_train.append(len(tracklet))
                    elif pid in selected_persons_test:
                        tracklets_test.append(tracklet)
                        num_imgs_per_tracklet_test.append(len(tracklet))
        # Filter out tracklets with fewer images than the specified minimum sequence length
        tracklets_train = [tracklet for tracklet in tracklets_train if len(tracklet) >= min_seq_len]
        tracklets_test = [tracklet for tracklet in tracklets_test if len(tracklet) >= min_seq_len]

        num_train_tracklets = len(tracklets_train)
        num_test_tracklets = len(tracklets_test)
        num_train_pids = len(selected_persons_train)
        num_test_pids = len(selected_persons_test)

        return tracklets_train, tracklets_test, num_train_tracklets, num_test_tracklets, \
                num_train_pids, num_test_pids, num_imgs_per_tracklet_train, num_imgs_per_tracklet_test


    def _create_query_gallery(self, tracklets):
        # gallery = defaultdict(dict)
        # query = defaultdict(dict)

        query = []
        gallery = []

        tracklet_query, tracklet_gallery = 0, 0    
        num_query_pids, num_gallery_pids = set(), set()

        for tracklet in tracklets:
            for img_path, person_id, camera_idx in tracklet:
                if camera_idx == 0:  # Camera A
                    #query[person_id][camera_idx] = img_path
                    query.append(tracklet)
                    tracklet_query += 1
                    num_query_pids.add(person_id)
                else:  # Cameras B and C
                    #gallery[person_id][camera_idx] = img_path
                    gallery.append(tracklet)
                    tracklet_gallery += 1
                    num_gallery_pids.add(person_id)

        num_query_pids =  len(list(num_query_pids))
        num_gallery_pids = len(list(num_gallery_pids))

        return query, gallery, tracklet_query, tracklet_gallery, num_query_pids, num_gallery_pids
        