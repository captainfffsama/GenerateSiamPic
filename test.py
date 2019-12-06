# -*- coding: utf-8 -*-

# @Description: 测试用的脚本
# @Author: CaptainHu
# @Date: 2019-12-06 15:24:51
# @LastEditTime: 2019-12-06 15:38:53
# @LastEditors: CaptainHu
import cv2

from dataset import XMLLikeDataset


xml_dir="/home/chiebotgpuhq/MyCode/dataset/Siam_detection/aqmzc"
a=XMLLikeDataset(xml_dir)
b,f,m=next(a)
cv2.imshow("b",b)
cv2.imshow("f",f)
cv2.imshow("m",m)
cv2.waitKey()