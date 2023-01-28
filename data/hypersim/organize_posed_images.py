import numpy as np
import os
from argparse import ArgumentParser
from functools import partial
import json
import shutil
import subprocess
from tqdm.contrib.concurrent import process_map


def get_scene_center(npz_path):
    """Get scene center from npz file."""
    data = np.load(npz_path)
    center = (data['bbox_min'] + data['bbox_max']) / 2

    # Convert from ngp to input space
    if not data['from_mitsuba']:
        center = center[[2, 0, 1]]

    center = (center - data['offset']) / data['scale']
    return center


def process_scene(path, npz_dir, limit, idx):
    """Process single Hypersim scene.

    Extract RGB images, poses and camera intrinsics.
    """
    output_path = os.path.join('posed_images', idx)
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    npz_path = os.path.join(npz_dir, idx + '.npz')
    center = get_scene_center(npz_path)
    
    img_dir = os.path.join(path, idx, 'train/images')
    
    xform_path = os.path.join(path, idx, 'train/transforms.json')
    with open(xform_path, 'r') as f:
        xforms = json.load(f)
    fl_x, fl_y, cx, cy = xforms['fl_x'], xforms['fl_y'], xforms['cx'], xforms['cy']
    intrinsic = np.array([[fl_x, 0, cx, 0],
                          [0, fl_y, cy, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    np.savetxt(os.path.join(output_path, 'intrinsic.txt'), intrinsic, fmt='%f')
    
    for frame in xforms['frames']:
        name = frame['file_path'].split('/')[-1][:-4]
        extrinsic = np.array(frame['transform_matrix'])
        extrinsic[:3, 3] -= center
        np.savetxt(os.path.join(output_path, f'{name}.txt'), extrinsic, fmt='%f')

        shutil.copy(os.path.join(img_dir, f'{name}.jpg'), os.path.join(output_path, f'{name}.jpg'))


def process_directory(path, npz_dir, limit, nproc):
    print(f'processing {path}')
    fn = partial(process_scene, path, npz_dir, limit)

    scene_names = os.listdir(npz_dir)
    scene_names = [name.split('.')[0] for name in scene_names]

    process_map(fn, sorted(scene_names), max_workers=nproc)
    
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max-images-per-scene', type=int, default=300)
    parser.add_argument('--nproc', type=int, default=8)
    args = parser.parse_args()

    process_directory('/data/bhuai/hypersim_results_all', 
                      '/data/bhuai/hypersim_rpn_data/features_200',
                      args.max_images_per_scene, args.nproc)
