# -*- coding: utf-8 -*-

# @Description: 图像增强
# @Author: HuQiong
# @Date: 2019-11-05 15:24:04
# @LastEditTime: 2019-11-19 12:18:50
# @LastEditors: HuQiong
import random

import cv2
from albumentations import(
    #通用
    OneOf,Compose,
    #几何
    #光学
    IAASharpen,IAAEmboss,RGBShift,HueSaturationValue,CLAHE,RandomContrast,RandomBrightness,
    RandomGamma
)

def shift_img(img,pad:int =50):
    h,w,_=img.shape
    img=cv2.copyMakeBorder(img,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    h_shift=random.randint(0,2*pad)
    w_shift=random.randint(0,2*pad)
    return img[h_shift:h_shift+h,w_shift:w_shift+w],pad-h_shift,pad-w_shift

class AGPolicy(object):

    @property
    def v1(self):
        seq=Compose([
            OneOf([
                CLAHE(),
                RandomGamma(),
            ],p=1),
            OneOf([
                RandomContrast(),
                RandomBrightness(),
            ],p=1)
        ])

        return seq



if __name__=="__main__":
    img=cv2.imread("/home/chiebotgpuhq/pic/2.jpg")
    img=shift_img(img)
    # cv2.imwrite('/home/chiebotgpuhq/Share/winshare/1.jpg',img_ag)
    ag=getattr(AGPolicy(),'v1')
    img=ag(image=img)['image']
    cv2.imshow("hah",img)
    cv2.waitKey()
