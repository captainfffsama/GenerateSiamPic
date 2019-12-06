# -*- coding: utf-8 -*-
# @Time    : 2019/06/10 18:56
# @Author  : CaptainHu

import os
import xml.etree.ElementTree as ET
from multiprocessing import Pool,freeze_support,cpu_count
import imghdr
import logging
import time
from contextlib import contextmanager

def get_all_xml_path(xml_dir:str,filter_=['.xml']):
    #遍历文件夹下所有xml
    result=[]
    #maindir是当前搜索的目录 subdir是当前目录下的文件夹名 file是目录下文件名
    for maindir,subdir,file_name_list in os.walk(xml_dir):
        for filename in file_name_list:
            ext=os.path.splitext(filename)[1]#返回扩展名
            if ext in filter_:
                result.append(os.path.join(maindir,filename))
    return result
    
def analysis_xml(xml_path:str):
    tree=ET.parse(xml_path)
    root=tree.getroot()
    result_dict={}
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        obj_num=result_dict.get(obj_name,0)+1
        result_dict[obj_name]=obj_num

    return result_dict

def analysis_xmls_batch(xmls_path_list:list):
    result_list=[]
    for i in xmls_path_list:
        if os.path.exists(i.replace('.xml','.JPG')):
            print(i.replace('.xml','.JPG'),'name is wrong')
            # logging.error(os.path.exists(i.replace('.xml','.JPG')),'name is wrong')
            os.rename(i.replace('.xml','.JPG'),i.replace('.xml','.jpg'))
        assert imghdr.what(i.replace('.xml','.jpg')) == 'jpeg','{} is worng'.format(i.replace('.xml','.jpg'))
        if not is_valid_jpg(i.replace('.xml','.jpg')):
            continue
        else:
            result_list.append(analysis_xml(i))
    return result_list

def collect_result(result_list:list):
    all_result_dict={}
    for result_dict in result_list:
        for key,values in result_dict.items():
            obj_num=all_result_dict.get(key,0)+values
            all_result_dict[key]=obj_num
    return all_result_dict

def main(xml_dir:'dir or txt',result_save_path:str =None):
    r'''根据xml文件统计所有样本的数目.对于文件不完整的图片和有xml但无图片的样本,直接进行删除.默认跑满所有的cpu核心
    
    Parameters
    ----------
    xml_dir : dir or txt
        dir:xml所在的文件夹.用的递归形式,因此只需保证xml在此目录的子目录下即可.对应的图片和其xml要在同一目录   

        txt:保存了xml所在的路径的txt
    
    result_save_path : str
        分析结果的日志保存路径.默认 None 无日志
    '''
    if result_save_path is not None:
        assert isinstance(result_save_path,str),'{} is illegal path'.format(result_save_path)
    else:
        logging.basicConfig(filename=result_save_path,filemode='w',level=logging.INFO)
    freeze_support()#windows 上用
    if os.path.isdir(xml_dir):
        xmls_path=get_all_xml_path(xml_dir)
    elif xml_dir.split('.')[-1]=='txt':
        with open(xml_dir,'r') as fr:
            xmls_path=[x.strip() for x in fr.readlines()]
    worker_num=cpu_count()
    print('your CPU num is',cpu_count())
    length=float(len(xmls_path))/float(worker_num)
    #计算下标，尽可能均匀地划分输入文件的列表
    indices=[int(round(i*length)) for i in range(worker_num+1)]
    

    #生成每个进程要处理的子文件列表
    sublists=[xmls_path[indices[i]:indices[i+1]] for i in range(worker_num)]
    pool=Pool(processes=worker_num)

    all_process_result_list=[]
    for i in range(worker_num):
        all_process_result_list.append(pool.apply_async(analysis_xmls_batch,args=(sublists[i],)))
    pool.close()
    pool.join()
    print('analysis done!')
    _temp_list=[]
    for i in all_process_result_list:
        _temp_list=_temp_list+i.get()
    result=collect_result(_temp_list)
    #logging.info(result)
    print(result)

def is_valid_jpg(jpg_file):
    """判断JPG文件下载是否完整     """
    if not os.path.exists(jpg_file):
        print(jpg_file,'is not existes')
        os.remove(jpg_file.replace('.jpg','.xml'))
        return False
    return True
    # with open(jpg_file, 'rb') as fr:
    #     fr.seek(-2, 2)
    #     if fr.read() == b'\xff\xd9':
    #         return True
    #     else:
    #         os.remove(jpg_file)
    #         os.remove(jpg_file.replace('.jpg','.xml'))
    #         print(jpg_file,'is imperfect img')
    #         logging.info(jpg_file,'is imperfect img')
    #         return False

if __name__=='__main__':
    test_dir='/home/chiebotgpuhq/MyCode/dataset/Siam_detection/aqmzc'
    save_path='/home/gpu-server/disk/disk1/game/temp_beijing_game_bc/dataset_analysis_result.log'
    main(test_dir)




