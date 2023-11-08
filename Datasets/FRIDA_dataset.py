import os
import json
from collections import defaultdict

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

        self.train, num_train_tracklets, num_train_pids, num_imgs_train = \
            self._process_data(self.train_dirs)
        
        num_imgs_per_tracklet = num_imgs_train
        min_num = min(num_imgs_per_tracklet)
        max_num = max(num_imgs_per_tracklet)
        avg_num = sum(num_imgs_per_tracklet) / num_train_tracklets

        num_total_pids = num_train_pids
        num_total_tracklets = num_train_tracklets

        print("=> FRIDA loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.num_train_pids = num_train_pids
        self.num_train_cams = len(self.cameras)
        self.num_train_vids = num_train_tracklets

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))

    def _process_data(self, dirnames):
        tracklets = []
        num_imgs_per_tracklet = []
        pid_container = set()

        for segment in dirnames:
            for camera in self.cameras:
                json_file = os.path.join(self.data_dir, 'Annotations', segment, camera, 'data2.json')
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for frame_info in data:
                    img_name = frame_info['file_name']
                    pid = frame_info['person_id']
                    pid_container.add(pid)
                    img_path = os.path.join(self.data_dir, 'BBs', segment, pid.zfill(8), camera, img_name)
                    tracklets.append((img_path, pid, self.cameras.index(camera)))
                    num_imgs_per_tracklet.append(1)

        num_tracklets = len(tracklets)
        num_pids = len(pid_container)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

# Example Usage:
# data_dir = 'path/to/FRIDA'
# frida = FRIDA(data_dir)
# train_data = frida.train
# num_train_pids = frida.num_train_pids
# num_train_cams = frida.num_train_cams
# num_train_vids = frida.num_train_vids
