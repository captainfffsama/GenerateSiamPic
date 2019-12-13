# -*- coding: utf-8 -*-

# @Description: 图像增强
# @Author: CaptainHu
# @Date: 2019-11-05 15:24:04
# @LastEditTime: 2019-12-11 12:17:17
# @LastEditors: CaptainHu
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

def rescale_img(bg,fg,mask):
    # 如果fg有一边大于bg  那就保持横纵比
    fg_hwrate=fg.shape[0]/fg.shape[1]
    if fg.shape[0] >bg.shape[0]:
        fg=cv2.resize(fg,None,fx=((bg.shape[1]-1)/fg.shape[0])/fg_hwrate,fy=(bg.shape[0]-1)/fg.shape[0])
        mask=cv2.resize(mask,None,fx=((bg.shape[1]-1)/mask.shape[0])/fg_hwrate,fy=(bg.shape[0]-1)/mask.shape[0])
    if fg.shape[1] >bg.shape[1]:
        fg=cv2.resize(fg,None,fx=((bg.shape[1]-1)/fg.shape[1]),fy=(bg.shape[0]-1)/fg.shape[1]*fg_hwrate)
        mask=cv2.resize(mask,None,fx=((bg.shape[1]-1)/mask.shape[1]),fy=(bg.shape[0]-1)/mask.shape[1]*fg_hwrate)
    return bg,fg,mask

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
