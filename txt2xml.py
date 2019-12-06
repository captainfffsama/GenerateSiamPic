# -*- coding: utf-8 -*-

# @Description: 之前的标签写错了，这个脚本将txt生成xml
# @Author: CaptainHu
# @Date: 2019-11-07 12:27:32
# @LastEditTime: 2019-11-07 13:44:40
# @LastEditors: CaptainHu
import os 
import glob    
from concurrent import futures

import cv2
from tqdm import tqdm

from utils.xml_tools import generate_xml,LabelInfo

def one_task(sample_dir):
    imgs_path_list=glob.glob(os.path.join(sample_dir,"*.jpg"))
    txt_path_list=glob.glob(os.path.join(sample_dir,"*.txt"))
    img_name=os.path.basename(imgs_path_list[0])
    img=cv2.imread(imgs_path_list[0])
    with open(txt_path_list[0],'r') as fr:
        content=fr.readlines()[0].strip()
    xmin,ymin,xmax,ymax=[int(x) for x in content.split(',')]
    sample_info=LabelInfo(img_name,
                          shape=img.shape,
                          objs_info=[("diff",xmin,ymin,xmax,ymax),])
    generate_xml(os.path.join(sample_dir,img_name.replace('.jpg','.xml')),
                 label_info=sample_info)

def main(dataset_dir:str):
    sample_dirs_list=glob.glob(os.path.join(dataset_dir,'*'))
    with futures.ThreadPoolExecutor(max_workers=64) as exec:
        tasks=[exec.submit(one_task,sample_dir) for sample_dir in sample_dirs_list]
        for task in tqdm(futures.as_completed(tasks),total=len(sample_dirs_list)):
            pass


        

if __name__=="__main__":
    sample_dir='/home/chiebotgpuhq/MyCode/dataset/Siam_detection/generate_sample'
    main(sample_dir)