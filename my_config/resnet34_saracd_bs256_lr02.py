"""
@ Author       : jonnyzhang 71881972+jonnyzhang02@users.noreply.github.com
@ LastEditTime : 2024-06-13 18:15
@ FilePath     : /mmpretrain/my_config/resnet34_saracd_bs256_lr02.py
@ 
@ coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn
"""
_base_ = [
    './datasets/SAR-ACD-sbs128.py',
    './models/resnet34.py',
    './schedules/bs256.py',
    './runtimes/runtime.py',
]