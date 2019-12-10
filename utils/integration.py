# -*- coding: utf-8 -*-

# @Description: 图片的融合方式
# @Author: CaptainHu
# @Date: 2019-12-10 15:20:31
# @LastEditTime: 2019-12-10 15:20:34
# @LastEditors: CaptainHu
import random

import cv2

def replace(bg,fg,mask):
    bg_h,bg_w,bg_c=bg.shape
    fg_h,fg_w,fg_c=fg.shape
    tl_point_x,tl_point_y=random.randint(0,bg_w-1-fg_w),random.randint(0,bg_h-1-fg_h)
    roi=(tl_point_x,tl_point_y,tl_point_x+fg_w,tl_point_y+fg_h)
    bg[roi[1]:roi[3],roi[0]:roi[2]]=fg
    return bg,roi
