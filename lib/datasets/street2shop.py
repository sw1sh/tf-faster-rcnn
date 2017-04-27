import os
import sys

import json
import pickle
import uuid

import numpy as np
import scipy.sparse

from datasets.imdb import imdb
from model.config import cfg

class street2shop(imdb):
  
  def __init__(self, image_set, data_path=None):
    imdb.__init__(self, 'street2shop_' + image_set)
    self._image_set = image_set

    self._data_path = self._get_default_path() if data_path is None else data_path
    self._classes = ('__background__',  # always index 0
                     'bags', 'belts', 'dresses', 'eyewear', 
                     'footwear', 'hats', 'leggings', 'outerwear',
                     'pants', 'skirts', 'tops')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.jpg'
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self._roidb_handler = self.selective_search_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    # specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': False,
                   'matlab_eval': False,
                   'rpn_file': None,
                   'min_size': 2}
  
  def _get_default_path(self):
    """
    Return the default path where is expected to be installed.
    """
    return os.path.join(cfg.DATA_DIR, 'street2shop')

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_path, 'images',
                              str(index) + self._image_ext)
    return image_path

  def _load_meta(self, class_):
    with open(os.path.join(self._data_path, 'meta', 'json',
                           self._image_set + '_pairs_' + class_ + '.json')) as f:
      return json.load(f)


  def _load_image_set_index(self):
    self._image_meta = {x['photo']: {'class': class_, 'bndbox': x['bbox']} 
                        for class_ in self._classes[1:] for x in self._load_meta(class_)}
    image_index = self._image_meta.keys()
    
  
    return filter(lambda index: os.path.exists(self.image_path_from_index(index)), image_index)

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._make_annotation(index)
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def selective_search_roidb(self):
    """
    Return the database of selective search regions of interest.
    Ground-truth ROIs are also included.
    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path,
                              self.name + '_selective_search_roidb.pkl')

    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        roidb = pickle.load(fid)
      print('{} ss roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    if self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      ss_roidb = self._load_selective_search_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
    else:
      roidb = self._load_selective_search_roidb(None)
    with open(cache_file, 'wb') as fid:
      pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote ss roidb to {}'.format(cache_file))

    return roidb

  def _make_annotation(self, index):

    meta = self._image_meta[index]
    bbox = meta['bndbox']
    
    x1 = bbox['left']
    y1 = bbox['top']
    x2 = x1 + bbox['width']
    y2 = y1 + bbox['height']
    
    box = [x1, y1, x2, y2]
    
    
    seg_area = bbox['width'] * bbox['height']
    
    overlaps = np.zeros((1, self.num_classes), dtype=np.float32)
    cls = self._class_to_ind[meta['class'] ]
    
    overlaps[0, cls] = 1.0
    
    overlaps = scipy.sparse.csr_matrix(overlaps)
       
    return {'boxes': np.asarray([box]),
            'gt_classes': np.asarray([cls]),
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': np.asarray([seg_area])}
