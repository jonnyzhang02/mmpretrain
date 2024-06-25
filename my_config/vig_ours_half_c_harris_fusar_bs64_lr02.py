"""
@ Author       : jonnyzhang 71881972+jonnyzhang02@users.noreply.github.com
@ LastEditTime : 2024-06-25 10:30
@ FilePath     : /mmpretrain/my_config/vig_ours_half_c_harris_fusar_bs64_lr02.py
@ 
@ coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn
"""
_base_ = [
    './datasets/FUSAR-sbs32.py',
    './models/vig_ours.py',
    './schedules/bs64.py',
    './runtimes/runtime.py',
]