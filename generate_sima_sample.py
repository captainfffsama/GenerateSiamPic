# -*- coding: utf-8 -*-

# @Description:对图像识别大赛中图片某以区域进行随机的变换 得到的训练集
# @Author: CaptainHu
# @Date: 2019-11-05 09:41:32
# @LastEditTime: 2019-12-16 10:18:50
# @LastEditors: CaptainHu

import os
import copy
import random
from concurrent import futures
import shutil

import cv2
from tqdm import tqdm

from utils.writer import save_xml
from utils.img_ag import AGPolicy,shift_img,rescale_img
from utils.integration import replace,seamlessclone
from dataset import XMLLikeDataset,COCODataset

class GenerateSiamsesSample(object):
    def __init__(self,dataset,save_dir:str,integration_mode:str = 'seamlessclone',img_ag:str='v1'):
        self._save_dir=save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.dataset=dataset
        if integration_mode=='seamlessclone':
            self.integration=seamlessclone
        elif integration_mode=='replace':
            self.integration=replace
        else:
            self.integration=seamlessclone
        if img_ag is not None:
            assert hasattr(AGPolicy(),img_ag),'there is no {} method,please check utils.img_ag.AGPolicy'.format(img_ag)
            self._img_ag=getattr(AGPolicy(),img_ag)
    
    def deal_one_sample(self,bg,fg,mask,idx):
        bg,fg,mask=rescale_img(bg,fg,mask)
        #TODO:
        #这里其实可以加个增强的步骤，暂时先不加
        try:
            img,box=self.integration(bg,fg,mask)
        except ValueError:
            print('idx',idx)
            return None 
        
        self._writer(idx,bg,img,box)

    def _writer(self,idx,img_o,img,box):
        '''
        img_o:原始图片，就是bg
        img:被P之后的结果图片
        box:P上的目标所在的位置(xmin,ymin,xmax,ymax)
        '''
        #初始化一些路径
        img_name=str(idx)+'.jpg'
        img_o_name=str(idx)+'_origin.jpg'
        save_dir=os.path.join(self._save_dir,str(idx))
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        #存图片
        cv2.imwrite(os.path.join(save_dir,img_name),img)
        cv2.imwrite(os.path.join(save_dir,img_o_name),img_o)
        save_xml(save_dir,idx,img,box)

    def test(self):
        bg,fg,mask=self.dataset[100]
        self.deal_one_sample(bg,fg,mask,100)

    def do_task(self,max_time:int=None,work_num:int =8):
        if max_time is None:
            max_time=len(self.dataset)
        '''
        XXX:测试代码
        '''
        for idx in tqdm(range(4589,max_time)):
            self.deal_one_sample(*self.dataset[idx],idx)
        # with futures.ThreadPoolExecutor() as exec:
        #     task_list=(exec.submit(self.deal_one_sample,*self.dataset[idx],idx) \
        #                 for idx in range(max_time))
        #     for task in tqdm(futures.as_completed(task_list),total=max_time):
        #         pass


if __name__=="__main__":
    json_path="/home/chiebotgpuhq/Share/gpu-server/disk/disk1/coco_dataset/annotations/instances_val2017.json"
    save_dir="/home/chiebotgpuhq/Share/gpu-server/disk/disk2/cocosiam_dataset/val"
    dataset=COCODataset(json_path,sampler='normal')
    test_cls=GenerateSiamsesSample(dataset,save_dir)
    test_cls.do_task()
    

