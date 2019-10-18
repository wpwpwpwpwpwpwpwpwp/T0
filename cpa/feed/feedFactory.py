# -*- coding: utf-8 -*-
'''
@Time    : 2019/7/20 22:25
@Author  : msi
@Email   : sdu.xuefu@gmail.com
'''

import platform
import os
import configparser as cg
from cpa.feed.baseFeed import PanelFeed

def platformSectionSelector():
    '''
    :return:数据路径选择
    '''
    if platform.system() == "Linux":
        return 'xuefu_linux'
    elif platform.node() == 'msi-PC':
        return 'xuefu_windows'
    else:
        return 'default'


class InlineDataSet:
    '''
    内部提供的测试集
    '''
    @classmethod
    def HS300_DAILY(cls):
        '''
        沪深300日频测试数据
        '''
        return dataFeedFactory.getHistFeed()



class dataFeedFactory:
    '''
    数据集中接口
    '''
    cfg = cg.ConfigParser()
    modulePath = os.path.dirname(os.path.abspath(__file__))
    cfg.read(os.path.join(modulePath, 'dataPath.ini'))
    DEFAULT_SECTION_PATH = platformSectionSelector()

    @classmethod
    def showLocalDataPath(cls):
        '''
        :return:显示本地数据存放路径
        '''
        return cls.DEFAULT_SECTION_PATH


    @classmethod
    def setConfigPath(cls, path, section=None):
        '''
        :param path:自定义配置文件路径
        :param section:
        :return:
        '''
        cls.cfg.read(os.path.join(path, 'dataPath.ini'))
        if section is not None:
            cls.DEFAULT_SECTION_PATH = section

    @classmethod
    def getHistFeed(cls, source='csv', instruments=None, start=None, end=None, maxLen=1024):
        '''
        :param source:本地数据源 csv h5等
        :param instruments: 加载的instrumets列表
        :param start: 起止时间
        :param end:
        :param maxLen: dataFeed缓存长度
        :return:
        '''

        data_path = cls.cfg.get(cls.DEFAULT_SECTION_PATH, source)
        print('reading data from path: {}'.format(data_path))
        panelFeed = None

        if source == 'csv':
            from cpa.dataReader import csvReader
            reader = csvReader.CSVReader(instruments, start=start)
            reader.set_file_path(path=data_path)  # 定义类
            reader.load_csv_files()  # 加载数据
            registerdInstruments = reader.getRegisteredInstruments()  # 获取实际拥有的instruments,可能有些数据没有csv
            panelFeed = PanelFeed(reader, registerdInstruments, maxLen=maxLen)

        elif source == 'h5':
            pass

        return panelFeed

    @classmethod
    def getLiveFeed(cls, source='tq', instruments=None, start=None, end=None, maxLen=1024):
        '''
        :param source: 实时数据流
        :param instruments:
        :param start:
        :param end:
        :param maxLen:
        :return:
        '''
        pass


if __name__ == '__main__':
    print(InlineDataSet.HS300_DAILY())