# -*- coding: utf-8 -*-

# @Description: 测试用的脚本
# @Author: CaptainHu
# @Date: 2019-12-06 15:24:51
# @LastEditTime: 2019-12-16 09:55:41
# @LastEditors: CaptainHu
import cv2
import matplotlib.pyplot as plt

import time
from contextlib import contextmanager
from dataset import XMLLikeDataset
from dataset import COCODataset
from generate_sima_sample import GenerateSiamsesSample
from utils.integration import seamlessclone
@contextmanager
def timeblock(label:str = '\033[1;34mSpend time:\033[0m'):
    r'''上下文管理测试代码块运行时间,需要
        import time
        from contextlib import contextmanager
    '''
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print('\033[1;34m{} : {}\033[0m'.format(label, end - start))

xml_dir="/home/chiebotgpuhq/MyCode/dataset/Siam_detection/aqmzc"
json_path="/home/chiebotgpuhq/Share/gpu-server/disk/disk1/coco_dataset/annotations/instances_val2017.json"
# a=XMLLikeDataset(xml_dir)
a=COCODataset(json_path)

# siam_g=GenerateSiamsesSample(a,"/home/chiebotgpuhq/MyCode/dataset/test")
# siam_g.test()
# with timeblock("too slow"):
b,f,m=a[1263]
# for b,f,m in a:
#     print(type(b))
# cv2.imshow("b",b)
# cv2.imshow("f",f)
# cv2.imshow("m",m)
# cv2.waitKey(100)
# b,roi=seamlessclone(b,f,m)
# plt.imshow(b)
plt.imshow(m)
# plt.imshow(m)
plt.show()
