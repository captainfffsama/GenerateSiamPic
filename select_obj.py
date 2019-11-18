# -*- coding: utf-8 -*-

# @Description: 从xml中选出要保留的目标 运行时间
# @Author: HuQiong
# @Date: 2019-07-02 10:47:10
# @LastEditTime: 2019-11-05 18:12:46
# @LastEditors: HuQiong

import xml.etree.ElementTree as ET
import os
import shutil
from tqdm import tqdm
import time
from contextlib import contextmanager
from pdb import set_trace
from concurrent import futures

def get_all_xml_path(xml_dir:str,filter_=['.xml']):
    #遍历文件夹下所有的xml
    result=[]
    for maindir,subdir,file_name_list in os.walk(xml_dir):
        for filename in file_name_list:
            ext=os.path.splitext(filename)[1]#返回扩展名
            if ext in filter_:
                result.append(os.path.join(maindir,filename))
    return result

def deal_xml(xml_path,savedir,classes=None,changelabel_dict=None):
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    xml_name=os.path.basename(xml_path)
    jpg_name=xml_name.replace('.xml','.jpg')
    tree=ET.parse(xml_path)
    root=tree.getroot()
    for obj in root.findall('object'):
        obj_name=obj.find('name').text
        if (changelabel_dict is not None) and (obj_name in changelabel_dict.keys()):
            obj.find('name').text=changelabel_dict[obj_name]
            continue
        if (classes is not None) and (obj_name not in classes):
            root.remove(obj)
            continue

    if root.find('object') is not None:
        tree.write(os.path.join(savedir,xml_name))
        shutil.copy(xml_path.replace('.xml','.jpg'),savedir)


@contextmanager
def timeblock(label:str):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print('{} : {}'.format(label, end - start))

def main(xml_dir,savedir,classes=None,changelabel_dict=None,work_num=64):
    r'''筛选并修改xml标签,注意修改之后的标签也会被都会被保存出来
    
    Parameters
    ----------
    xmldir : (str)
        xml文件所在目录，会搜索子目录
    
    savedir : (str)
        筛选和修改之后的数据存放的目录，会将修改之后的标签和图片拷贝过来
    
    classes : (tuple = None)
        指定了 哪些标签会被保留，若为None，则所有标签都会保留
    
    changelabel_dict : (dict = None)
        键表示需要修改的标签名，值修改之后的标签名，默认为None就是所有标签不修改

    work_num: (int = 64)
        最大线程数量
    '''
    xmls_path=get_all_xml_path(xml_dir)
    with futures.ThreadPoolExecutor(work_num) as pool:
        task_list=(pool.submit(deal_xml,xml_path,savedir,classes,changelabel_dict) for xml_path in xmls_path)
        for x in tqdm(futures.as_completed(task_list),total=len(xmls_path)):
            pass




if __name__=='__main__':
    with timeblock('time'):
        xml_dir='/home/chiebotgpuhq/MyCode/dataset/Siam_detection/train'
        savedir='/home/chiebotgpuhq/MyCode/dataset/Siam_detection/aqmzc'
        classes=('aqmzc')
        changelabel_dict={
                'no_helmet':'wcaqm',
                'have_helmet':'aqmzc',
                'wcgc':'wcgz'
                # 'dz_czjgx':'xtsb',
                # 'dlq_czx':'xtsb',
                # 'zhdq_jgx':'xtsb',
                # 'byq_tg':'jyz',
                # 'cqtg_bt':'jyz',
                # 'zhdq_tg':'jyz'
        }
        main(xml_dir,savedir,classes)



