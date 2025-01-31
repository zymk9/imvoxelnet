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

COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}

COMPRESSION_TYPE_DEPTH = {
    -1: 'unknown',
    0: 'raw_ushort',
    1: 'zlib_ushort',
    2: 'occi_ushort'
}


class RGBDFrame:
    """Class for single ScanNet RGB-D image processing."""

    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack('f' * 16, file_handle.read(16 * 4)),
            dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = b''.join(
            struct.unpack('c' * self.color_size_bytes,
                          file_handle.read(self.color_size_bytes)))
        self.depth_data = b''.join(
            struct.unpack('c' * self.depth_size_bytes,
                          file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        assert compression_type == 'zlib_ushort'
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        assert compression_type == 'jpeg'
        return imageio.imread(self.color_data)


class SensorData:
    """Class for single ScanNet scene processing.

    Single scene file contains multiple RGB-D images.
    """

    def __init__(self, filename, limit):
        self.version = 4
        self.load(filename, limit)

    def load(self, filename, limit):
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = b''.join(
                struct.unpack('c' * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack(
                'i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack(
                'i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            num_frames = struct.unpack('Q', f.read(8))[0]
            self.frames = []
            if limit > 0 and limit < num_frames:
                index = np.random.choice(
                    np.arange(num_frames), limit, replace=False).tolist()
            else:
                index = list(range(num_frames))
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                if i in index:
                    self.frames.append(frame)

    def export_depth_images(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for f in range(len(self.frames)):
            depth_data = self.frames[f].decompress_depth(
                self.depth_compression_type)
            depth = np.fromstring(
                depth_data, dtype=np.uint16).reshape(self.depth_height,
                                                     self.depth_width)
            imageio.imwrite(
                os.path.join(output_path,
                             self.index_to_str(f) + '.png'), depth)

    def export_color_images(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for f in range(len(self.frames)):
            color = self.frames[f].decompress_color(
                self.color_compression_type)
            imageio.imwrite(
                os.path.join(output_path,
                             self.index_to_str(f) + '.jpg'), color)

    @staticmethod
    def index_to_str(index):
        return str(index).zfill(5)

    @staticmethod
    def save_mat_to_file(matrix, filename):
        with open(filename, 'w') as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt='%f')

    def export_poses(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for f in range(len(self.frames)):
            self.save_mat_to_file(
                self.frames[f].camera_to_world,
                os.path.join(output_path,
                             self.index_to_str(f) + '.txt'))

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.save_mat_to_file(self.intrinsic_color,
                              os.path.join(output_path, 'intrinsic.txt'))


def process_scene(path, limit, idx):
    """Process single ScanNet scene.

    Extract RGB images, poses and camera intrinsics.
    """
    output_path = os.path.join('posed_images', idx)
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    img_dir = os.path.join(path, idx, 'train/images')
    subprocess.run('cp {}/* {}/'.format(img_dir, output_path), shell=True)
    
    xform_path = os.path.join(path, idx, 'train/transforms.json')
    with open(xform_path, 'r') as f:
        xforms = json.load(f)
    fl_x, fl_y, cx, cy = xforms['fl_x'], xforms['fl_y'], xforms['cx'], xforms['cy']
    intrinsic = np.array([[fl_x, 0, cx],
                          [0, fl_y, cy],
                          [0, 0, 1]])
    np.savetxt(os.path.join(output_path, 'intrinsic.txt'), intrinsic)
    
    for frame in xforms['frames']:
        name = frame['file_path'].split('/')[-1].split('.')[0]
        extrinsic = np.array(frame['transform_matrix'])
        np.savetxt(os.path.join(output_path, f'{name}.txt'), extrinsic)



def process_directory(path, limit, nproc):
    print(f'processing {path}')
    mmcv.track_parallel_progress(
        func=partial(process_scene, path, limit),
        tasks=sorted(os.listdir(path)),
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
    parser.add_argument('--max-images-per-scene', type=int, default=1000)
    parser.add_argument('--nproc', type=int, default=8)
    args = parser.parse_args()

    process_directory('FRONT3D_render/finished', args.max_images_per_scene, args.nproc)
