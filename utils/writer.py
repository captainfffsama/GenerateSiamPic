# -*- coding: utf-8 -*-

# @Description: 一些结果保存方式
# @Author: CaptainHu
# @Date: 2019-12-10 17:16:31
# @LastEditTime: 2019-12-10 18:40:15
# @LastEditors: CaptainHu
import os

from .xml_tools import LabelInfo,analysis_label_info,generate_xml

def save_xml(save_dir,idx,img,box):
    save_img_info=LabelInfo(os.path.join(save_dir,str(idx)+".jpg"),\
                            shape=img.shape,
                            objs_info=[("diff",*box),])
    generate_xml(os.path.join(save_dir,str(idx)+".xml"),save_img_info)