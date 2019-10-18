# -*- coding: utf-8 -*-
'''
@Time    : 2019/6/8 23:01
@Author  : msi
@Email   : sdu.xuefu@gmail.com
临时存放，后期可以使用配置文件
'''
import platform

if platform.system() == "Linux":
    path = '/root/PycharmProjects/t0_frameWork/cpa/data/'
elif platform.node() == 'msi-PC':
    path = 'D:/output/t0_framework_data/'
else:
    path =r'F:/wangpeng/dist/example/data'#数据存放路径

#临时存放缠论数据

if platform.system() == "Linux":
    chanlun_data_path = 'F:/wangpeng/data/zz500_stock20180601to20190601.h5'
elif platform.node() == 'msi-PC':
    chanlun_data_path = 'C:/Users/msi/PycharmProjects/t0_frameWork/chanlun/data/zz500_stock20180601to20190601.h5'
else:
    chanlun_data_path = r'C:\Users\lixiao\PycharmProjects\t0_frameWork\chanlun\dataSource\zz500_stock20180601to20190601.h5'  # 数据存放路径
