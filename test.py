# -*- coding: utf-8 -*-

# @Description: 测试用的脚本
# @Author: CaptainHu
# @Date: 2019-12-06 15:24:51
# @LastEditTime: 2019-12-10 14:15:28
# @LastEditors: CaptainHu
import cv2

from dataset import XMLLikeDataset
from dataset import COCODataset

xml_dir="/home/chiebotgpuhq/MyCode/dataset/Siam_detection/aqmzc"
json_path="/home/chiebotgpuhq/Share/gpu-server/disk/disk1/coco_dataset/annotations/instances_train2017.json"
# a=XMLLikeDataset(xml_dir)
a=COCODataset(json_path)
b,f,m=next(a)
for b,f,m in a:
    cv2.imshow("b",b)
    cv2.imshow("f",f)
    cv2.imshow("m",m)
    cv2.waitKey(0)
