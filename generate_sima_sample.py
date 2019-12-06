# -*- coding: utf-8 -*-

# @Description:对图像识别大赛中图片某以区域进行随机的变换 得到的训练集
# @Author: CaptainHu
# @Date: 2019-11-05 09:41:32
# @LastEditTime: 2019-12-05 16:40:39
# @LastEditors: CaptainHu

import os
import copy
import random
from concurrent import futures
import shutil

import cv2
from tqdm import tqdm

from utils.xml_tools import LabelInfo,analysis_label_info,generate_xml
from utils.img_ag import AGPolicy,shift_img

class GenerateSiamsesSample(object):
    def __init__(self,xml_dir:str,save_dir:str,img_ag:str='v1'):
        self._save_dir=save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        assert os.path.exists(xml_dir),'Error! {} is not exist'.format(xml_dir)
        self._xml_dir=xml_dir
        self._all_xml_path_list=self.get_all_xml_path(xml_dir)
        if img_ag is not None:
            assert hasattr(AGPolicy(),img_ag),'there is no {} method,please check utils.img_ag.AGPolicy'.format(img_ag)
            self._img_ag=getattr(AGPolicy(),img_ag)
    

    def deal_one_sample(self,idx,xml_path:str):
        #取得要粘贴的目标和背景图片
        #两者预处理
        #融合方法（seamlessclone和直接贴）
        #保存
        sample_info=analysis_label_info(xml_path)
        if len(sample_info.objs_info)==0:
            return
        _sample_save_path=os.path.join(self._save_dir,str(idx))
        if os.path.exists(_sample_save_path):
            shutil.rmtree(_sample_save_path)
        os.makedirs(_sample_save_path)
        img=cv2.imread(os.path.join(self._xml_dir,sample_info.jpg_name))
        img_o=copy.deepcopy(img)
        obj=sample_info.objs_info[0]
        
        xmin,ymin,xmax,ymax=obj[1:]
        roi_w,roi_h=xmax-xmin,ymax-ymin
        img_h,img_w,_=sample_info.shape

        tl_point_x,tl_point_y=random.randint(0,img_w-1-roi_w),random.randint(0,img_h-1-roi_h)
        roi=(tl_point_x,tl_point_y,tl_point_x+roi_w,tl_point_y+roi_h)
        obj_img=img[roi[1]:roi[3],roi[0]:roi[2],:]
        img[ymin:ymax,xmin:xmax]=obj_img

        img,h_shift,w_shift=shift_img(img,0)
        xmin=max(0,xmin+w_shift)
        xmax=min(xmax+w_shift,img.shape[1])
        ymin=max(0,ymin+h_shift)
        ymax=min(ymax+h_shift,img.shape[0])
        if hasattr(self,'_img_ag'):
            img=self._img_ag(image=img)['image']
        #存样本
        cv2.imwrite(os.path.join(_sample_save_path,str(idx)+".jpg"),img)
        cv2.imwrite(os.path.join(_sample_save_path,str(idx)+"_original.jpg"),img_o)
        save_img_info=LabelInfo(os.path.join(_sample_save_path,str(idx)+".jpg"),\
                                shape=sample_info.shape,
                                objs_info=[("diff",xmin,ymin,xmax,ymax),])
        generate_xml(os.path.join(_sample_save_path,str(idx)+".xml"),save_img_info)

    def do_task(self):
        # self._all_xml_path_list=['/home/chiebotgpuhq/MyCode/dataset/Siam_detection/aqmzc/ffcd422486d5c6fc0d12604457737e59.xml',]
        # for idx,xml_path in tqdm(enumerate(self._all_xml_path_list)):
        #    self.deal_one_sample(idx,xml_path)
        with futures.ProcessPoolExecutor() as exec:
            task_list=(exec.submit(self.deal_one_sample,idx,xml_path) \
                        for idx,xml_path in enumerate(self._all_xml_path_list))
            for task in tqdm(futures.as_completed(task_list),total=len(self._all_xml_path_list)):
                pass


if __name__=="__main__":
    xml_dir="/home/chiebotgpuhq/MyCode/dataset/Siam_detection/aqmzc"
    save_dir="/home/chiebotgpuhq/MyCode/dataset/Siam_detection/gen_siam"
    test_cls=GenerateSiamsesSample(xml_dir,save_dir)
    # print(test_cls._all_xml_path_list)    
    test_cls.do_task()
    # test_xml="/home/chiebotgpuhq/MyCode/dataset/Siam_detection/train/ffff28d3380f99b27de35c0dc6478849.xml"
    # test_cls.deal_one_sample(0,test_xml)

