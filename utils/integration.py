# -*- coding: utf-8 -*-

# @Description: 图片的融合方式
# @Author: CaptainHu
# @Date: 2019-12-10 15:20:31
# @LastEditTime: 2019-12-12 19:43:22
# @LastEditors: CaptainHu
import random
from copy import deepcopy

import cv2

def replace(bg,fg,mask):
    bg_t=deepcopy(bg)
    bg_t_h,bg_t_w,bg_t_c=bg_t.shape
    fg_h,fg_w,fg_c=fg.shape
    tl_point_x,tl_point_y=random.randint(0,bg_t_w1-fg_w),random.randint(0,bg_t_h-fg_h)
    roi=(tl_point_x,tl_point_y,tl_point_x+fg_w,tl_point_y+fg_h)
    bg_t[roi[1]:roi[3],roi[0]:roi[2]]=fg
    return bg_t,roi

def seamlessclone(bg,fg,mask):
    bg_h,bg_w,bg_c=bg.shape
    fg_h,fg_w,fg_c=fg.shape
    tl_point_x,tl_point_y=random.randint(0,bg_w-fg_w),random.randint(0,bg_h-fg_h)
    c_x,c_y=tl_point_x+fg_w//2,tl_point_y+fg_h//2
    roi=(tl_point_x,tl_point_y,tl_point_x+fg_w,tl_point_y+fg_h)
    try:
        result=cv2.seamlessClone(fg,bg,mask,(c_x,c_y),cv2.NORMAL_CLONE)
    except cv2.error:
        raise ValueError('opencv error')
    return result,roi
