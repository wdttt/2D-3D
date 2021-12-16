import glob
import json
import pandas
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import seaborn
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import mayavi.mlab
import torch.nn.functional as F
from torch.autograd import Variable
import os.path as osp
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, project=False, H=64, W=2048, fov_up=3.0, fov_down=-25.0):
    self.project = project
    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.reset()

  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)

    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                            dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                            dtype=np.int32)

    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                              dtype=np.int32)       # [H,W] mask

  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, filename):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission
    self.set_points(points, remissions)

  def set_points(self, points, remissions=None):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    # put in attribute
    self.points = points    # get xyz
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

    # if projection is wanted, then do it and fill in the structure
    if self.project:
      self.do_range_projection()

  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # laser parameters
    fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1)

    # get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= self.proj_W                              # in [0.0, W]
    proj_y *= self.proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    self.proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    self.proj_y = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order
    self.unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    self.proj_range[proj_y, proj_x] = depth
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remission[proj_y, proj_x] = remission
    self.proj_idx[proj_y, proj_x] = indices
    self.proj_mask = (self.proj_idx > 0).astype(np.int32)
class SemLaserScan(LaserScan):
  """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
  EXTENSIONS_LABEL = ['.label']

  def __init__(self,  sem_color_dict=None, project=False, H=64, W=2048, fov_up=3.0, fov_down=-25.0, max_classes=300):
    super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down)
    self.reset()

    # make semantic colors
    if sem_color_dict:
      # if I have a dict, make it
      max_sem_key = 0
      for key, data in sem_color_dict.items():
        if key + 1 > max_sem_key:
          max_sem_key = key + 1
      self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
      for key, value in sem_color_dict.items():
        self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
    else:
      # otherwise make random
      max_sem_key = max_classes
      self.sem_color_lut = np.random.uniform(low=0.0,
                                             high=1.0,
                                             size=(max_sem_key, 3))
      # force zero to a gray-ish color
      self.sem_color_lut[0] = np.full((3), 0.1)

    # make instance colors
    max_inst_id = 100000
    self.inst_color_lut = np.random.uniform(low=0.0,
                                            high=1.0,
                                            size=(max_inst_id, 3))
    # force zero to a gray-ish color
    self.inst_color_lut[0] = np.full((3), 0.1)

  def reset(self):
    """ Reset scan members. """
    super(SemLaserScan, self).reset()

    # semantic labels
    self.sem_label = np.zeros((0, 1), dtype=np.int32)          # [m, 1]: label
    self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # instance labels
    self.inst_label = np.zeros((0, 1), dtype=np.int32)          # [m, 1]: label
    self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # projection color with semantic labels
    self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                   dtype=np.int32)              # [H,W]  label
    self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                   dtype=np.float64)              # [H,W,3] color

    # projection color with instance labels
    self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                                    dtype=np.int32)              # [H,W]  label
    self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
                                    dtype=np.float64)              # [H,W,3] color

  def open_label(self, filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
      raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.int32)
    label = label.reshape((-1))

    # set it
    self.set_label(label)

  def set_label(self, label):
    """ Set points for label not from file but from np
    """
    # check label makes sense
    if not isinstance(label, np.ndarray):
      raise TypeError("Label should be numpy array")

    # only fill in attribute if the right size
    if label.shape[0] == self.points.shape[0]:
      self.sem_label = label & 0xFFFF  # semantic label in lower half
      self.inst_label = label >> 16    # instance id in upper half
    else:
      print("Points shape: ", self.points.shape)
      print("Label shape: ", label.shape)
      raise ValueError("Scan and Label don't contain same number of points")

    # sanity check
    assert((self.sem_label + (self.inst_label << 16) == label).all())

    if self.project:
      self.do_label_projection()

  def colorize(self):
    """ Colorize pointcloud with the color of each semantic label
    """
    self.sem_label_color = self.sem_color_lut[self.sem_label]
    self.sem_label_color = self.sem_label_color.reshape((-1, 3))

    self.inst_label_color = self.inst_color_lut[self.inst_label]
    self.inst_label_color = self.inst_label_color.reshape((-1, 3))

  def do_label_projection(self):
    # only map colors to labels that exist
    mask = self.proj_idx >= 0

    # semantics
    self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
    self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]

    # instances
    self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
    self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]
class DummyDataset(Dataset):
  """Use torch dataloader for multiprocessing"""

  def __init__(self):
    self.data = []
    self.glob_frames()

  def glob_frames(self):
    glob_path = '../../Desktop/000000.png'
    cam_paths = sorted(glob.glob(glob_path))
    # load calibration
    calib = self.read_calib('../../Desktop/calib.txt')
    proj_matrix = calib['P2'] @ calib['Tr']
    proj_matrix = proj_matrix.astype(np.float32)

    for cam_path in cam_paths:
      basename = osp.basename(cam_path)
      frame_id = osp.splitext(basename)[0]
      assert frame_id.isdigit()
      data = {
        'camera_path': cam_path,
        'lidar_path': '../../Desktop/000000.bin',
        'label_path': '../../Desktop/000000.label',
        'proj_matrix': proj_matrix
      }
      for k, v in data.items():
        if isinstance(v, str):
          if not osp.exists(v):
            raise IOError('File not found {}'.format(v))
      self.data.append(data)

  @staticmethod
  def read_calib(calib_path):
    """
    :param calib_path: Path to a calibration text file.
    :return: dict with calibration matrices.
    """
    calib_all = {}
    with open(calib_path, 'r') as f:
      for line in f.readlines():
        if line == '\n':
          break
        key, value = line.split(':', 1)
        calib_all[key] = np.array([float(x) for x in value.split()])

    # reshape matrices
    calib_out = {}
    calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
    calib_out['Tr'] = np.identity(4)  # 4x4 matrix
    calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)
    return calib_out

  @staticmethod
  def select_points_in_frustum(points_2d, x1, y1, x2, y2):
    """
    Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
    :param points_2d: point cloud projected into 2D
    :param points_3d: point cloud
    :param x1: left bound
    :param y1: upper bound
    :param x2: right bound
    :param y2: lower bound
    :return: points (2D and 3D) that are in the frustum
    """
    keep_ind = (points_2d[:, 0] > x1) * \
               (points_2d[:, 1] > y1) * \
               (points_2d[:, 0] < x2) * \
               (points_2d[:, 1] < y2)

    return keep_ind

  def __getitem__(self, index):
    data_dict = self.data[index].copy()
    scan = np.fromfile(data_dict['lidar_path'], dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, :3]
    label = np.fromfile(data_dict['label_path'], dtype=np.uint32)
    label = label.reshape((-1))
    label = label & 0xFFFF  # get lower half for semantics

    # load image
    image = Image.open(data_dict['camera_path'])
    image_size = image.size

    # project points into image
    keep_idx = points[:, 0] > 0  # only keep point in front of the vehicle
    points_hcoords = np.concatenate([points[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
    img_points = (data_dict['proj_matrix'] @ points_hcoords.T).T
    all_img_points = (data_dict['proj_matrix'] @ np.concatenate([points, np.ones([len(points), 1], dtype=np.float32)], axis=1).T).T
    img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
    all_img_points = all_img_points[:, :2] / np.expand_dims(all_img_points[:, 2], axis=1)  # scale 2D points
    keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, *image_size)
    keep_idx[keep_idx] = keep_idx_img_pts
    # fliplr so that indexing is row, col and not col, row
    img_points = np.fliplr(img_points)
    all_img_points = np.fliplr(all_img_points)
    depth = np.linalg.norm(points, 2, axis=1)
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    # debug
    # from xmuda.data.utils.visualize import draw_points_image, draw_bird_eye_view
    # draw_points_image(np.array(image), img_points[keep_idx_img_pts].astype(int), label[keep_idx],
    #                   color_palette_type='SemanticKITTI_long')

    data_dict['seg_label'] = label[keep_idx].astype(np.int16)
    data_dict['points'] = points[keep_idx]
    data_dict['points_img'] = img_points[keep_idx_img_pts]
    data_dict['image_size'] = np.array(image_size)
    data_dict['scan'] = scan[keep_idx]
    data_dict['depth'] = depth[keep_idx]
    data_dict['all_points_img'] = all_img_points
    data_dict['all_seg_label'] = label.astype(np.int16)
    data_dict['image'] = image

    return data_dict

  def __len__(self):
    return len(self.data)
da = DummyDataset()
color_map={
  0 : [0, 0, 0],
  1 : [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0]}


import torch
from torch import nn
from torch.nn import functional as F
import torchvision


import argparse
print(np.array([1,2,3]))
'''
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scales',help='scales',type=int,default=5)
    opt = parser.parse_args()
    return opt
if __name__ == '__main__':
    opt = parse()
    print(opt)
    opt.num = 'now two args'
    print(opt)
    '''
'''
def do_range_projection(points,seg_label,color_map):
  """ Project a pointcloud into a spherical projection image.projection.
      Function takes no arguments because it can be also called externally
      if the value of the constructor was not set (in case you change your
      mind about wanting the projection)
  """
  # laser parameters
  H = 64
  W = 2048
  fov_up = 3.0
  fov_down = -25.0
  fov_up = fov_up / 180.0 * np.pi  # field of view up in rad
  fov_down =fov_down / 180.0 * np.pi  # field of view down in rad
  fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

  # get depth of all points
  depth = np.linalg.norm(points, 2, axis=1)

  # get scan components
  scan_x = points[:, 0]
  scan_y = points[:, 1]
  scan_z = points[:, 2]

  # get angles of all points
  yaw = -np.arctan2(scan_y, scan_x)
  pitch = np.arcsin(scan_z / depth)

  # get projections in image coords
  proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
  proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

  # scale to image size using angular resolution
  proj_x *= W  # in [0.0, W]
  proj_y *= H  # in [0.0, H]

  # round and clamp for use as index
  proj_x = np.floor(proj_x)
  proj_x = np.minimum(W - 1, proj_x)
  proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
  proj_x = np.copy(proj_x)  # store a copy in orig order

  proj_y = np.floor(proj_y)
  proj_y = np.minimum(H - 1, proj_y)
  proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
  proj_y = np.copy(proj_y)  # stope a copy in original order

  # copy of depth in original order
  unproj_range = np.copy(depth)

  # order in decreasing depth
  indices = np.arange(depth.shape[0])
  order = np.argsort(depth)[::-1]
  points = points[order]
  seg_label = seg_label[order]
  proj_y = proj_y[order]
  proj_x = proj_x[order]
  proj_xyz = np.full((H, W, 3), -1,
                     dtype=np.float32)
  proj_label= np.full((H, W,3), -1,
                     dtype=np.float32)

  # assing to images
  proj_xyz[proj_y, proj_x] = points
  proj_label[proj_y, proj_x] = [color_map[seg_label[i]] for i in range(len(seg_label))]
  return proj_xyz,proj_label
for i in da:
  seg_label=i['seg_label']
  points = i['points']
  points_img = i['points_img']
  image_size = i['image_size']
  #image_size = np.array([2048,376])
  all_points_img = i['all_points_img']
  all_seg_label = i['all_seg_label']
  image = i['image']
  depth = i['depth']
  proj_remission = np.full((image_size[1], image_size[0]), -1, dtype=np.float32)
  proj_label = np.full((image_size[1], image_size[0],3), -1, dtype=np.int32)
  proj_depth = np.full((image_size[1], image_size[0]), -1, dtype=np.float32)
  for j in range(len(points_img)):
    proj_y=points_img[j,0]
    proj_x=points_img[j,1]
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(image_size[0] - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
    proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(image_size[1] - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
    proj_y = np.copy(proj_y)  # stope a copy in original order
    #proj_remission[proj_y, proj_x] = i['scan'][j,3]
    proj_label[proj_y, proj_x,:] = color_map[seg_label[j]]
    #proj_depth[proj_y, proj_x] = depth[j]
  print('seg_label:',seg_label.shape)
  print('points:', points.shape)
  print('points_img:', points_img.shape)
  print('image_size:', image_size.shape)
  print('all_points_img:', all_points_img.shape)
  print('all_seg_label:',all_seg_label.shape)
  #plt.imshow(np.concatenate([image, proj_label], axis=0))

  (point,label) = do_range_projection(points,seg_label,color_map)
  plt.imshow(label)
  plt.show()
'''


'''
color_map={
  0 : [0, 0, 0],
  1 : [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0]}
scan = SemLaserScan(project=True,sem_color_dict=color_map)

scan.open_scan('../../Desktop/000000.bin')
scan.open_label('../../Desktop/000000.label')

proj_range = torch.from_numpy(scan.proj_range).clone()
proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
proj_remission = torch.from_numpy(scan.proj_remission).clone()
proj_mask = torch.from_numpy(scan.proj_mask)
proj_x = torch.full([150000], -1, dtype=torch.long)
proj_x[:scan.points.shape[0]] = torch.from_numpy(scan.proj_x)
proj_y = torch.full([150000], -1, dtype=torch.long)
proj_y[:scan.points.shape[0]] = torch.from_numpy(scan.proj_y)
proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2,0,1),
                      proj_remission.unsqueeze(0).clone()])
proj = proj * proj_mask.float()
print("points:",scan.points.shape)
print("proj_x:",scan.proj_x.shape)
print("proj_y:",scan.proj_y.shape)
print("proj_H:",scan.proj_H)
print("proj_W:",scan.proj_W)
print("proj_range:",scan.proj_range.shape)
print("proj_remission:",scan.proj_remission.shape)
print("sem_label:",scan.sem_label.shape)
print("proj_sem_color:",scan.proj_sem_color.shape)
print("proj_sem_label:",scan.proj_sem_label.shape)
print("proj_inst_color:",scan.proj_inst_color.shape)
print("proj_inst_label:",scan.proj_inst_label.shape)
plt.rcParams['figure.dpi'] = 300 #分辨率
plt.imshow(scan.proj_sem_color)
plt.show()
'''








'''
class ShapeNetDataset(Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}
        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split/', 'shuffled_{}_file_list.json'.format(split))
        # from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid + '.pts'),
                                                         os.path.join(self.root, category, 'points_label',
                                                                      uuid + '.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(self.root, 'num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        # print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)
        '''
'''
class ModelNetDataset(Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='modelnet40_train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())
        self.cat = {}
        with open('../../Datasets/modelnet40_normal_resampled/modelnet40_shape_names.txt', 'r') as f:
            for idx,line in enumerate(f,0):
                self.cat[line.replace('\n','')] = idx

        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        if(fn.split('_')[0] in self.cat.keys()):
            cls = self.cat[fn.split('_')[0]]
            root1 = fn.split('_')[0]
        else:
            cls = self.cat[fn.split('_')[0]+'_'+fn.split('_')[1]]
            root1 = fn.split('_')[0]+'_'+fn.split('_')[1]
        with open(self.root+'/'+root1+'/'+fn+'.txt', 'r') as f:
            point = f.read()
            l1 = point.replace('\n', ',')
            l2 = l1.split(',')
            l2.pop()
            m1 = np.array(l2[0:60000])
            m2 = m1.reshape(10000, 6)
            x = [float(k[0]) for k in m2]
            y = [float(k[1]) for k in m2]
            z = [float(k[2]) for k in m2]
        pts = np.vstack([x, y, z]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter
        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


    def __len__(self):
        return len(self.fns)

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[1]
        x = x.transpose(2, 1)
        trans = self.stn(x)
        #x = x.transpose(2, 1)
        x = torch.bmm(trans,x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.transpose(2, 1)
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = x.transpose(2, 1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x.transpose(2, 1)
        x = torch.max(x, 1, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024,1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat.transpose(2, 1)], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=40, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.k = k
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[1]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = x.view(batchsize, n_pts, self.k)
        return x

traindataset = ShapeNetDataset(root='../../Datasets/shapenetcore_partanno_segmentation_benchmark_v0/')
traindataloader = DataLoader(traindataset,shuffle=True,batch_size=256)
net = PointNetCls()
numepoch = 5
learning_rate = 0.01
weight_decay = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
def train():
    for epoch in range(numepoch):
        train_loss =[]
        train_miou = []
        #train_accs = []
        for X,y in traindataloader:
            y_pred = net(X)
            print(y_pred.shape,y.shape)
            #y = np.squeeze(y)
            loss = 0
            for i in range(2500):
              loss =loss + criterion(y_pred[:,i,:],y[:,i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #acc = (y_pred.argmax(dim=-1) == y).float().mean()
            miou = (y_pred.argmax(dim=-1) == y).float().mean()
            # Record the loss and accuracy.
            train_loss.append(int(loss.item()*100)/100)
            train_miou.append(int(miou.item()*10000)/10000)
            print('trainloss',train_loss)
            print('trainmiou',train_miou)
        train_loss = sum(train_loss) / len(train_loss)
        #train_acc = sum(train_accs) / len(train_accs)
        train_miou = sum(train_miou) / len(train_miou)
        print('loss',train_loss)
        #print('acc',train_acc)
if __name__ == '__main__':
    train()
'''
'''
f = open('../../Datasets/courtyard_dslr_jpg/courtyard/dslr_calibration_jpg/points3D.txt', 'r')
point = f.read()
f.close()
l1 = point.split('\n')
l1 = l1[3:30491]
l2 = np.array([k.split(' ')[1:7] for k in l1])
# 数据预处理
l1 = point.replace('\n', ' ')
# 将数据以“，”进行分割
l2 = l1.split(' ')
l2.pop()
# print(l2)
# 将数据转为矩阵

m1 = np.array(l2[0:7000000])
# 变形
m2 = m1.reshape(-1, 7)
m2 = l2
x = [float(k[0]) for k in m2]
y = [float(k[1]) for k in m2]
z = [float(k[2]) for k in m2]
r = [int(k[3]) for k in m2]
g = [int(k[4]) for k in m2]
b = [int(k[5]) for k in m2]
a = [255 for k in m2]
rgba = np.concatenate((np.array(r).reshape(-1,1),np.array(g).reshape(-1,1),np.array(b).reshape(-1,1),np.array(a).reshape(-1,1)),axis=1)
pts = mayavi.mlab.pipeline.scalar_scatter(x, y, z) # plot the points
pts.add_attribute(rgba, 'colors') # assign the colors to each point
pts.data.point_data.set_active_scalars('colors')
g = mayavi.mlab.pipeline.glyph(pts)
g.glyph.glyph.scale_factor = 0.5 # set scaling for all the points
g.glyph.scale_mode = 'data_scaling_off' # make all the points same size


#f = mayavi.mlab.points3d(x, y, z,(r,g,b),scale_factor = .005)
mayavi.mlab.show()
'''
'''
labels_dataframe = pd.read_csv('../../Datasets/classify-leaves/train.csv')
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(labels_dataframe)
class_to_num = dict(zip(leaves_labels, range(n_classes)))
num_to_class = {v : k for k, v in class_to_num.items()}

# 继承pytorch的dataset，创建自己的
class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """

        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode

        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path, header=None)  # header=None是去掉表头部分
        # 计算 length
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            # 第一列包含图像文件的名称
            self.train_image = np.asarray(
                self.data_info.iloc[1:self.train_len, 0])  # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            # 第二列是图像的 label
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name)

        # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        #         if img_as_img.mode != 'L':
        #             img_as_img = img_as_img.convert('L')

        # 设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
                transforms.ToTensor()
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            # number label
            number_label = class_to_num[label]

            return img_as_img, number_label  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len
train_path = '../../Datasets/classify-leaves/train.csv'
test_path = '../../Datasets/classify-leaves/test.csv'
# csv文件中已经images的路径了，因此这里只到上一级目录
img_path = '../../Datasets/classify-leaves/'

train_dataset = LeavesData(train_path, img_path, mode='train')
val_dataset = LeavesData(train_path, img_path, mode='valid')
test_dataset = LeavesData(test_path, img_path, mode='test')
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=5
    )

val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=5
    )
test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=5
    )
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False
# resnet34模型
def res_model(num_classes, feature_extract = False, use_pretrained=True):

    model_ft = torchvision.models.resnet34(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft
learning_rate = 3e-4
weight_decay = 1e-3
num_epoch = 50
model_path = './pre_res_model.ckpt'

model = res_model(176)
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# The number of training epochs.
n_epochs = num_epoch

best_acc = 0.0
if __name__ == '__main__':
    for epoch in range(n_epochs):
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()
        # These are used to record information in training.
        train_loss = []
        train_accs = []
        # Iterate the training set by batches.
        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            imgs = imgs
            labels = labels
            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs)
            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels)

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()
            # Compute the gradients for parameters.
            loss.backward()
            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()
        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(val_loader):
            imgs, labels = batch
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs)

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels)

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # if the model improves, save a checkpoint at this epoch
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc))

'''
'''
train_data = pd.read_csv('../../Datasets/kaggle_house/train.csv')
test_data = pd.read_csv('../../Datasets/kaggle_house/test.csv')
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features = pd.get_dummies(all_features,dummy_na=True)
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values).to(torch.float32)
test_features = torch.tensor(all_features[n_train:].values).to(torch.float32)
train_labels = torch.tensor(train_data['SalePrice'].values).to(torch.float32).view(-1,1)
print(train_features.shape,train_labels.shape)

Batch_size = 256
dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_loader = DataLoader(dataset,batch_size=Batch_size,shuffle=True)
class Linearnet(nn.Module):
    def __init__(self):
        super(Linearnet, self).__init__()
        self.linear1 = nn.Linear(all_features.shape[1],all_features.shape[1]//2)
        self.linear2 = nn.Linear(all_features.shape[1] // 2,1)
    def forward(self,x):
        x=self.linear2(self.linear1(x))
        return x
loss = nn.MSELoss()

def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()

def train(num_epochs,train_features, train_labels, test_features, test_labels):
    train_ls,test_ls = [],[]
    net = Linearnet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        for X,y in train_loader:
            y_pred = net(X)
            ls = loss(y_pred,y)
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid
def k_fold(k, X_train, y_train, num_epochs):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        train_ls, valid_ls = train(num_epochs, *data)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k
train_l, valid_l = k_fold(5, train_features, train_labels, 1)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (5, train_l, valid_l))
train_ls,_ =train(1,train_features, train_labels,None,None)
print('train rmse %f' % train_ls[-1])
'''
