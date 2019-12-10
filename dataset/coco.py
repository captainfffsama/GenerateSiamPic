# -*- coding: utf-8 -*-

# @Description: 使用coco合成
# @Author: CaptainHu
# @Date: 2019-12-09 19:20:38
# @LastEditTime: 2019-12-10 13:48:44
# @LastEditors: CaptainHu
import random

from pycocotools.coco import COCO
import skimage.io as io
import numpy as np

from .basic_dataset import BasicDataset

class COCODataset(BasicDataset):
    def __init__(self,json_path:str, cats:list = None,sampler='random'):
        self._coco=COCO(json_path)
        self._cats=cats
        cats=self._coco.loadCats(self._coco.getCatIds())
        self.sampler=self._get_sampler(sampler)
        self._imgIDs=self._coco.getImgIds()

    def set_cats(cats:list = None):
        self._cats=cats

    def __len__(self):
        return len(self._imgIDs)

    def _get_bg(self,idx):
        imgs=self._coco.loadImgs(self._imgIDs[idx])
        return io.imread(imgs[0]["coco_url"])[:,:,::-1]

    def _get_fg(self):
        if self._cats is not None and isinstance(self._cats,Iterable):
            catsIDs=self._coco.getCatIds(catNms=self._cats)
            imgIds=self._coco.getImgIds(catIds=catsIDs)
        else:
            imgIds=self._coco.getImgIds()
        imgs=self._coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])
        fg=io.imread(imgs[0]["coco_url"])[:,:,::-1]
        annIds = self._coco.getAnnIds(imgIds=imgs[0]['id'], iscrowd=None)
        anns = self._coco.loadAnns(annIds)
        mask=self._coco.annToMask(anns[0])
        mask=mask*255
        return fg,mask