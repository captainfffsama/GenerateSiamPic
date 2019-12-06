# -*- coding: utf-8 -*-

# @Description: 数据集类的基类，虚类，不要实例化
# @Author: CaptainHu
# @Date: 2019-12-06 13:20:04
# @LastEditTime: 2019-12-06 15:38:32
# @LastEditors: CaptainHu
import os
import random

class BasicDataset(object):
    def __init__(self,sampler:str = 'random'):
        self.sampler=self._get_sampler(sampler)
        self._idx=0
        raise AttributeError("You must over write __init__,and set sampler and _idx")

    def __len__(self):
        raise AttributeError("You must overwrite __len__")

    def __next__(self):
        return self.sampler(self)

    def _get_sampler(self,sampler_name:str):

        def random_(self):
            idx=random.randint(0,len(self)-1)
            return self[idx]
            
        def normal_(self):
            idx=self._idx
            self._idx=self._idx+1 if self._idx+1<len(self) else 0
            return self[idx]

        if sampler_name == "random":
            return random_
        elif sampler_name == "normal":
            return normal_
        else:
            print("Oh,here is no sampler named {},bro,please add it in sampler function".format(sampler_name))
            print("I will use random sampler")
            return random_
    
    def __getitem__(self,idx):
        bg=self._get_bg(idx)
        fg,mask=self._get_fg()
        return bg,fg,mask 

    def _get_bg(self,idx):
        raise AttributeError("You must overwrite _get_bg(self,idx),and return a pic")

    def _get_fg(self):
        raise AttributeError("You must overwrite _get_fg(self),and return a pic and it mask")
