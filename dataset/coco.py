# -*- coding: utf-8 -*-

# @Description: 使用coco合成
# @Author: CaptainHu
# @Date: 2019-12-09 19:20:38
# @LastEditTime: 2019-12-11 10:36:23
# @LastEditors: CaptainHu
import random
import math
import os

from pycocotools.coco import COCO
import cv2
import numpy as np

from .basic_dataset import BasicDataset

from ipdb import set_trace

import time
from contextlib import contextmanager

@contextmanager
def timeblock(label:str = '\033[1;34mSpend time:\033[0m'):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print('\033[1;34m{} : {}\033[0m'.format(label, end - start))

'''
NOTE:注意COCODataset虽然支持序列的一些特性，单不包含切片特性
'''
class COCODataset(BasicDataset):
    def __init__(self,json_path:str, pic_dir:str = None, cats:list = None,sampler='random'):
        self._coco=COCO(json_path)
        self._cats=cats
        self.sampler=self._get_sampler(sampler)
        self._imgIDs=self._coco.getImgIds()
        if pic_dir is not None:
            self._pic_dir=pic_dir
        else:
            data_name=os.path.basename(json_path).split('.')[0].split('_')[-1]
            self._pic_dir=os.path.realpath(os.path.join(os.path.dirname(json_path),'..',data_name))
        print('COCO init done!!!')

    def _imread(self,imgID):
        imgs_info=self._coco.loadImgs(imgID)
        return cv2.imread(os.path.join(self._pic_dir,imgs_info[0]['file_name'])),imgs_info[0]

    def set_cats(self,cats:list = None):
        self._cats=cats

    def __len__(self):
        return len(self._imgIDs)

    def _get_bg(self,idx):
        result,_=self._imread(self._imgIDs[idx])
        return result

    def _get_fg(self):
        if self._cats is not None and isinstance(self._cats,Iterable):
            catsIDs=self._coco.getCatIds(catNms=self._cats)
            imgIds=self._coco.getImgIds(catIds=catsIDs)
        else:
            imgIds=self._imgIDs
        fg,img_info=self._imread(random.choice(imgIds))
        annIds = self._coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
        anns = self._coco.loadAnns(annIds)
        mask=self._coco.annToMask(anns[0])*255
        box=self._deal_limit(mask.shape,anns[0]["bbox"])
        mask=mask[box[1]:box[3],box[0]:box[2]]
        fg=fg[box[1]:box[3],box[0]:box[2]]
        return fg,mask
    
    #NOTE:注意COCO原始box是(x,y,w,h)，这里变换完之后变成tr和bl
    def _deal_limit(self,img_shape,box):
        box_=[1]*4
        box_[0]=max(0,round(box[0]))
        box_[1]=max(0,round(box[1]))
        box_[2]=min(img_shape[1]-1,round(box[0]+box[2]))
        box_[3]=min(img_shape[0]-1,round(box[1]+box[3]))
        return box_ 