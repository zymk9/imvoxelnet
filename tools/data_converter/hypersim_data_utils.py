import mmcv
import numpy as np
import os
from concurrent import futures as futures
from os import path as osp


class HypersimData(object):
    """Hypersim data.
    Generate Hypersim infos for Hypersim converter.
    Args:
        root_path (str): Root path of the raw data.
        split (str): Set split type of the data. Default: 'train'.
    """

    def __init__(self, root_path, split='train'):
        self.root_dir = root_path
        self.split = split
        self.split_dir = osp.join(root_path)
        self.classes = ['object']
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids = np.array([0])
        self.cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }

        assert split in ['train', 'val', 'test']
        split_file = osp.join(self.root_dir, 'meta_data',
                              f'hypersim_rpn_{split}.txt')
        mmcv.check_file_exist(split_file)
        self.sample_id_list = mmcv.list_from_file(split_file)
        self.test_mode = (split == 'test')

    def __len__(self):
        return len(self.sample_id_list)

    def get_aligned_box_label(self, idx):
        box_file = osp.join(self.root_dir, 'hypersim_instance_data',
                            f'{idx}_aligned_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_unaligned_box_label(self, idx):
        box_file = osp.join(self.root_dir, 'hypersim_instance_data',
                            f'{idx}_unaligned_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_axis_align_matrix(self, idx):
        matrix_file = osp.join(self.root_dir, 'hypersim_instance_data',
                               f'{idx}_axis_align_matrix.npy')
        mmcv.check_file_exist(matrix_file)
        return np.load(matrix_file)

    def get_images(self, idx):
        paths = []
        path = osp.join(self.root_dir, 'posed_images', idx)
        for file in sorted(os.listdir(path)):
            if file.endswith('.jpg'):
                paths.append(osp.join('posed_images', idx, file))
        return paths

    def get_extrinsics(self, idx):
        extrinsics = []
        path = osp.join(self.root_dir, 'posed_images', idx)
        for file in sorted(os.listdir(path)):
            if file.endswith('.txt') and not file == 'intrinsic.txt':
                extrinsics.append(np.loadtxt(osp.join(path, file)))
        return extrinsics

    def get_intrinsics(self, idx):
        matrix_file = osp.join(self.root_dir, 'posed_images', idx,
                               'intrinsic.txt')
        mmcv.check_file_exist(matrix_file)
        return np.loadtxt(matrix_file)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.
        This method gets information from the raw data.
        Args:
            num_workers (int): Number of threads to be used. Default: 4.
            has_label (bool): Whether the data has label. Default: True.
            sample_id_list (list[int]): Index list of the sample.
                Default: None.
        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()

            # update with RGB image paths if exist
            if os.path.exists(osp.join(self.root_dir, 'posed_images')):
                info['intrinsics'] = self.get_intrinsics(sample_idx)
                info['extrinsics'] = self.get_extrinsics(sample_idx)
                info['img_paths'] = self.get_images(sample_idx)

                assert len(info['img_paths']) == len(info['extrinsics']), \
                    f'Number of images and extrinsics do not match for ' \
                    f'sample {sample_idx}'

            if has_label:
                annotations = {}
                # box is of shape [k, 7 + class]
                aligned_box_label = self.get_aligned_box_label(sample_idx)
                annotations['gt_num'] = aligned_box_label.shape[0]
                if annotations['gt_num'] != 0:
                    aligned_box = aligned_box_label[:, :-1]  # k, 7
                    classes = aligned_box_label[:, -1]  # k
                    annotations['name'] = np.array([
                        'object' for i in range(annotations['gt_num'])
                    ])
                    # default names are given to aligned bbox for compatibility
                    # we also save unaligned bbox info with marked names
                    annotations['location'] = aligned_box[:, :3]
                    annotations['dimensions'] = aligned_box[:, 3:6]
                    annotations['gt_boxes_upright_depth'] = aligned_box
                    annotations['index'] = np.arange(
                        annotations['gt_num'], dtype=np.int32)
                    annotations['class'] = np.zeros(annotations['gt_num'])

                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
