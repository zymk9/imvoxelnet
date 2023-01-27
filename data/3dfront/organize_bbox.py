# Modified from https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py # noqa
import imageio
import mmcv
import numpy as np
import os
import struct
import zlib
from argparse import ArgumentParser
from functools import partial
import json
import shutil
import subprocess


def process_bbox(path, idx):
    """Process a single 3D-FRONT scene.
    
    Organize bounding boxes and labels.
    """
    output_path = os.path.join('3dfront_instance_data')
    os.makedirs(output_path, exist_ok=True)
    
    xform_path = os.path.join(path, idx, 'train/transforms.json')
    with open(xform_path, 'r') as f:
        xforms = json.load(f)

    obbs = []
    for bbox in xforms['bounding_boxes']:
        center = np.array(bbox['position'])
        size = np.array(bbox['extents'])
        orientation = np.array(bbox['orientation'])
        angle = np.arctan2(orientation[1,0], orientation[0,0])
        obb = np.concatenate([center, size, np.array([angle, 0])]) # [x, y, z, w, l, h, theta, class]
        obbs.append(obb)
    obbs = np.stack(obbs, axis=0)
    np.save(os.path.join(output_path, '{}_aligned_bbox.npy'.format(idx)), obbs)


def process_directory(path, nproc):
    print(f'processing {path}')
    mmcv.track_parallel_progress(
        func=partial(process_bbox, path),
        tasks=[x.split('.')[0] for x in sorted(os.listdir(path))],
        nproc=nproc)

def prep_data_split(path, outdir):
    data_split = dict(np.load(path))
    os.makedirs(outdir, exist_ok=True)
    splits = ['train', 'val', 'test']
    for split_prefix in splits:
        with open(os.path.join(outdir, "3dfront_rpn_{}.txt".format(split_prefix)), 'w') as f:
            for scene in sorted(data_split['{}_scenes'.format(split_prefix)]):
                f.write(scene)
                f.write('\n')
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nproc', type=int, default=8)
    args = parser.parse_args()

    process_directory('FRONT3D_render/finished', args.nproc)
