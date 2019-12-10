# -*- coding: utf-8 -*-

# @Description: 用于管理类似voc使用xml管理的dataset
# @Author: CaptainHu
# @Date: 2019-12-05 17:00:04
# @LastEditTime: 2019-12-09 19:19:46
# @LastEditors: CaptainHu
import os
import random

import cv2
import numpy as np

from .basic_dataset import BasicDataset
from utils.xml_tools import analysis_label_info

'''
xml数据集返回的是随机裁减出来的一张目标作为前景,返回的mask和前景box一样大小
'''
class XMLLikeDataset(BasicDataset):
    def __init__(self,xml_dir:str,sampler:str = 'random'):
        self._all_xml_path=self._get_all_xml_path(xml_dir)
        self.sampler=self._get_sampler(sampler)

    def _get_all_xml_path(self,xml_dir:str,filter_=('.xml')) -> list: 
        #遍历文件夹下所有的xml
        return [os.path.join(maindir,filename) for maindir,_,file_name_list in os.walk(xml_dir) \
            for filename in file_name_list \
            if os.path.splitext(filename)[1] in filter_ ]

    def __len__(self):
        return len(self._all_xml_path)

    def _get_bg(self, idx):
        bg_pic_path=self._all_xml_path[idx].replace(".xml",".jpg")
        bg_pic=cv2.imread(bg_pic_path)
        return bg_pic
    
    def _get_obj_info(self):
        idx=random.randint(0,len(self)-1)
        xml_path=self._all_xml_path[idx]
        return xml_path,analysis_label_info(xml_path)

    def _get_fg(self):
        while True:
            xml_path,sample_info=self._get_obj_info()
            if len(sample_info.objs_info) != 0:
                break
        obj=sample_info.objs_info[0]
        xmin,ymin,xmax,ymax=obj[1:]
        fg_whole_path=xml_path.replace(".xml",".jpg")
        fg_whole_pic=cv2.imread(fg_whole_path)
        fg_pic=fg_whole_pic[ymin:ymax,xmin:xmax,:]
        mask=np.ones_like(fg_pic)*255
        return fg_pic,mask
    

