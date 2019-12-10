# -*- coding: utf-8 -*-

# @Description: 测试用的脚本
# @Author: CaptainHu
# @Date: 2019-12-06 15:24:51
# @LastEditTime: 2019-12-10 16:50:08
# @LastEditors: CaptainHu
import cv2

from dataset import XMLLikeDataset
from dataset import COCODataset
from generate_sima_sample import GenerateSiamsesSample

xml_dir="/home/chiebotgpuhq/MyCode/dataset/Siam_detection/aqmzc"
json_path="/home/chiebotgpuhq/Share/gpu-server/disk/disk1/coco_dataset/annotations/instances_train2017.json"
# a=XMLLikeDataset(xml_dir)
a=COCODataset(json_path)
siam_g=GenerateSiamsesSample(a,None)
img,box=siam_g.test()
cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),thickness=3)
print(box)
cv2.imshow("img",img)
cv2.waitKey(0)
# b,f,m=next(a)
# for b,f,m in a:
#     cv2.imshow("b",b)
#     cv2.imshow("f",f)
#     cv2.imshow("m",m)
#     cv2.waitKey(100)
