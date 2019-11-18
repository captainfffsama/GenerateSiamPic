# -*- coding: utf-8 -*-

# @Description: 图像增强
# @Author: HuQiong
# @Date: 2019-11-05 15:24:04
# @LastEditTime: 2019-11-08 13:34:55
# @LastEditors: HuQiong

import imgaug.augmenters as iaa
import cv2

class Policy(object):

    @property
    def v1(self):
        pol= iaa.Sequential([
                iaa.SomeOf((1,2),
                [
                    iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.75, 2.0)),
                    iaa.Emboss(alpha=(0.0, 0.5), strength=(0.5, 1.5)),
                    iaa.HistogramEqualization(),
                    iaa.Multiply((0.5, 1.5)),
                    iaa.Add((-30, 30),per_channel=0.3),
                ],random_order=True)])

        return pol

if __name__=="__main__":
    img=cv2.imread("/home/chiebotgpuhq/MyCode/dataset/Siam_detection/train/0a38dc97f2afbd6383972cd82a8f3ff6.jpg")
    img_ag=getattr(Policy(),"v1")(img)
    cv2.imshow("hah",img_ag)
    cv2.waitKey()