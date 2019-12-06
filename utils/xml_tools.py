# -*- coding: utf-8 -*-

# @Description: xml常用的一些工具
# @Author: CaptainHu
# @Date: 2019-11-05 11:02:08
# @LastEditTime: 2019-11-07 15:45:49
# @LastEditors: CaptainHu
import xml.etree.ElementTree as ET
import os

class LabelInfo(object):
    def __init__(self,jpg_name:str=None,shape:tuple=None,objs_info:list=None):
        r'''
            用于保存图片的标签信息
        
            Parameters
            ----------
            jpg_name : (str)
                仅仅只是jpg文件的名称+后缀名
        
            shape : (tuple)
                h，w，c
        
            objs_info : (list[('label_name',xmin,ymin,xmax,ymax)])
                目标的信息，保存方式是('label_name',xmin,ymin,xmax,ymax)
        '''
        self._jpg_name=jpg_name
        self._shape=shape
        self._objs_info=[] if objs_info is None else objs_info
    
    @property
    def jpg_name(self):
        return self._jpg_name

    @jpg_name.setter
    def jpg_name(self,value):
        if not isinstance(value,str):
            raise TypeError("jpg_name must be a str")
        self._jpg_name=value
    
    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self,value):
        if not (isinstance(value,tuple) and 3==len(value)):
            raise TypeError("shape must be tuple and contain h,w,c")
        self._shape=value
    
    @property
    def objs_info(self):
        return self._objs_info
    
    @objs_info.setter
    def objs_info(self,value):
        if not isinstance(value,list):
            raise TypeError("objs_info  must be list")
        self._objs_info=value
    
    def add_obj(self,obj_info:tuple):
        if not (isinstance(obj_info,tuple) and isinstance(obj_info[0],str)):
            raise TypeError("obj_info must be ('label_name',xmin,ymin,xmax,ymax)")
        self._objs_info.append(obj_info)

    def __str__(self):
        return "\n".join(['-'*20,"jpg:{}".format(self._jpg_name),
                        "shape:{}".format(str(self._shape)),\
            "objs:",*[str(obj_info) for obj_info in self._objs_info],'='*20])


def analysis_label_info(xml_path:str) -> "LabelInfo":
    r'''分析标签的信息，将xml信息组织成LabelInfo
        Parameters
        ----------
        xml_path : (str)
            xml的路径
        Outputs:
        ----------
        - **LabelInfo**: 包含了sample信息
    '''
    jpg_name=os.path.basename(xml_path).replace('xml','jpg')
    tree=ET.parse(xml_path)
    root=tree.getroot()
    
    size_xml=root.find('size')
    size=(int(size_xml.find('height').text), \
          int(size_xml.find('width').text), \
          int(size_xml.find('depth').text))
    
    objs_info_list=[]
    for obj in root.findall('object'):
        obj_name=obj.find('name').text
        bbox_element=obj.find('bndbox')
        xmin=int(float(bbox_element.find('xmin').text))
        ymin=int(float(bbox_element.find('ymin').text))
        xmax=int(float(bbox_element.find('xmax').text))
        ymax=int(float(bbox_element.find('ymax').text))
        objs_info_list.append((obj_name,xmin,ymin,xmax,ymax))

    return LabelInfo(jpg_name,size,objs_info_list)

def generate_xml(xml_path,label_info:"LabelInfo"):
    #形成xml
    root=ET.Element('annotation')
    e_folder=ET.SubElement(root,'folder')
    e_folder.text=os.path.dirname(xml_path)
    e_filename=ET.SubElement(root,'filename')
    e_filename.text=os.path.basename(xml_path.replace('.xml','.jpg'))
    e_path=ET.SubElement(root,'path')
    e_path.text=xml_path
    e_source=ET.SubElement(root,'source')
    ee_database=ET.SubElement(e_source,'database')
    ee_database.text='Unknown'

    #获取图片信息
    h,w,c=label_info.shape

    e_size=ET.SubElement(root,'size')
    ee_width=ET.SubElement(e_size,'width')
    ee_height=ET.SubElement(e_size,'height')
    ee_depth=ET.SubElement(e_size,'depth')
    ee_width.text=str(w)
    ee_height.text=str(h)
    ee_depth.text=str(c)

    e_segmented=ET.SubElement(root,'segmented')
    e_segmented.text='0'

    #写入obj信息
    if len(label_info.objs_info)!=0:
        for obj in label_info.objs_info:
            e_object=ET.SubElement(root,'object')
            ee_name=ET.SubElement(e_object,'name')
            ee_name.text=obj[0]
            ee_pose=ET.SubElement(e_object,'pose')
            ee_pose.text='Unspecified'
            ee_truncated=ET.SubElement(e_object,'truncated')
            ee_truncated.text='0'
            ee_difficult=ET.SubElement(e_object,'difficult')
            ee_difficult.text='0'
            ee_bndbox=ET.SubElement(e_object,'bndbox')
            eee_xmin=ET.SubElement(ee_bndbox,'xmin')
            eee_ymin=ET.SubElement(ee_bndbox,'ymin')
            eee_xmax=ET.SubElement(ee_bndbox,'xmax')
            eee_ymax=ET.SubElement(ee_bndbox,'ymax')
            eee_xmin.text=str(obj[1])
            eee_ymin.text=str(obj[2])
            eee_xmax.text=str(obj[3])
            eee_ymax.text=str(obj[4])

    _beautify_xml_root(root)

    # ET.dump(root)
    tree=ET.ElementTree(root)
    tree.write(xml_path)

def _beautify_xml_root(elem,level=0):
    i='\r\n'+level*'\t'
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text=i+'\t'
        if not elem.tail or not elem.tail.strip():
            elem.tail=i
        if level ==0:
            elem.tail=''
        for elem in elem:
            _beautify_xml_root(elem,level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail =i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail=i


if __name__=="__main__":
    a=analysis_label_info('/home/chiebotgpuhq/MyCode/dataset/Siam_detection/train/ffff28d3380f99b27de35c0dc6478849.xml')
    print(a)
    generate_xml("/home/chiebotgpuhq/test.xml",a)
    print("save done!")