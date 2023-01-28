import numpy as np
import os
from argparse import ArgumentParser
from functools import partial
import json
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


def process_bbox(xform_dir, npz_dir, scene_name):
    """Process a single Hypersim scene.
    
    Organize bounding boxes and labels. Assumes that the xform_dir contains
    ai_xxx_xxx.json which contains the object filter info.
    """
    output_path = 'hypersim_instance_data'
    os.makedirs(output_path, exist_ok=True)

    xform_path = os.path.join(xform_dir, scene_name + '.json')
    npz_path = os.path.join(npz_dir, scene_name + '.npz')

    scene_center = get_scene_center(npz_path)
    
    with open(xform_path, 'r') as f:
        xforms = json.load(f)

    obbs = []
    for bbox in xforms['bounding_boxes']:
        if bbox['filtered']:
            continue        # consistent with NeRF RPN dataset
        center = np.array(bbox['position']) - scene_center
        size = np.array(bbox['extents'])
        orientation = np.array(bbox['orientation'])
        angle = np.arctan2(orientation[1,0], orientation[0,0])
        obb = np.concatenate([center, size, np.array([angle, 0])]) # [x, y, z, w, l, h, theta, class]
        obbs.append(obb)

    if len(obbs) == 0:
        obbs = np.zeros((0, 8))
    else:
        obbs = np.stack(obbs, axis=0)
        
    np.save(os.path.join(output_path, '{}_aligned_bbox.npy'.format(scene_name)), obbs)


def process_directory(xform_dir, npz_dir, nproc):
    print(f'processing {xform_dir}')
    scene_names = os.listdir(npz_dir)
    scene_names = [x.split('.')[0] for x in scene_names]

    fn = partial(process_bbox, xform_dir, npz_dir)

    process_map(fn, sorted(scene_names), max_workers=nproc)


def prep_data_split(path, outdir):
    data_split = dict(np.load(path))
    os.makedirs(outdir, exist_ok=True)
    splits = ['train', 'val', 'test']
    for split_prefix in splits:
        with open(os.path.join(outdir, "hypersim_rpn_{}.txt".format(split_prefix)), 'w') as f:
            for scene in sorted(data_split['{}_scenes'.format(split_prefix)]):
                f.write(scene)
                f.write('\n')

        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nproc', type=int, default=8)
    args = parser.parse_args()

    process_directory('/data/bhuai/hypersim_rpn_data/obb_filtering_results', 
                      '/data/bhuai/hypersim_rpn_data/features_200',
                      args.nproc)
    prep_data_split('/data/bhuai/hypersim_rpn_data/hypersim_split_new.npz', 'meta_data')
