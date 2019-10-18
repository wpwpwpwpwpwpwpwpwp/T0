import os

import numpy as np
import pandas as pd


def read_csv_with_filter(path, instrument, start=None):
    '''
    读取csv数据并转换为DataFrame,包括open,high,low,close,volume,turnover,date,code
    :param path: 路径
    :param instrument:代码
    :param start: 起始时间
    :return: df, 读取csv文件
    '''
    fileName = '{}.csv'.format(instrument)
    filePath = os.path.join(path, fileName)
    df = pd.read_csv(filePath, header=None)
    df.columns = ['day', 'time', 'open', 'high', 'low', 'close', 'volume', 'turnover']
    df['date'] = pd.to_datetime(df['day'] + ' ' + df['time'])
    df['code'] = instrument
    if start is not None:
        df = df[df['date'] > start]
    return df[['code', 'date', 'open', 'high', 'low', 'close', 'volume']]


class CSVReader:
    '''
    从本地数据库读取股票日行情数据  
    :return: 每运行一次getNextValues则输出一个时间截面所有股票数据df，行为代码，列为开高低收量信息
    '''

    def __init__(self, instruments=None, start=None):
        self.instruments = instruments
        self.start = pd.to_datetime(start)
        self.dfs = []
        self.isEof = False
        self.valGen = None

    def _iter_(self):
        return self

    def set_file_path(self, path):
        '''
        :param path: 设置文件读取路径
        :return:
        '''
        self.path = path

    def get_file_list(self):
        '''
        :return:获取路径下的所有csv文件
        '''
        fileLists = os.listdir(self.path)  # 读取文件夹下所有文件名
        ret = []
        for file in fileLists:
            if '.csv' in file.lower():
                ret.append(file.split('.')[0])
        return ret

    def load_csv_files(self):
        '''
        :return: 从文件路径中读取相应的csv文件
        '''
        fileLists = self.get_file_list()
        if self.instruments is None:
            self.instruments = fileLists
        removeList = list(set(self.instruments) - set(fileLists))  # 求差集
        if len(removeList) > 0:
            print('{} removed'.format(removeList))
        self.instruments = list(set(self.instruments) & set(fileLists))  # 求交集
        self.instruments.sort()
        self.codeFrame = pd.DataFrame({'code': self.instruments})

        for instrument in self.instruments:
            print('reading file {}'.format(instrument))
            thisData = read_csv_with_filter(self.path, instrument, self.start)
            self.dfs.append(thisData)
        self.dfs = pd.concat(self.dfs, axis=0)
        self.dfs.sort_values(['date', 'code'], ascending=True, inplace=True)
        self.allTime = np.unique(self.dfs['date'])

        self.valGen = self.valueGenerator()

    def valueGenerator(self):
        '''
        :return:从dfs中读取下一个时间的数据,返回时间和一个dataframe：行为code,列为高开低收量，{下一时刻时间:对应股票数据dataframe}
        '''
        for idx, _date in enumerate(self.allTime):
            ret = self.dfs[self.dfs['date'] == _date]
            ret = pd.merge(ret, self.codeFrame, on='code', how='outer')
            ret = ret.set_index('code')
            del ret['date']
            yield pd.to_datetime(_date), ret

            if idx == len(self.allTime) - 2:
                self.isEof = True

    def getNextValues(self):
        return next(self.valGen)

    def getRegisteredInstruments(self):
        '''
        :return: 返回实际使用的instruments
        '''
        return self.instruments

    def eof(self):
        '''
        :return: 如果没有数据更新,返回True;有数据更新返回False，并更新self.start的值
        '''
        return self.isEof


if __name__ == '__main__':
    from cpa.utils.pathSelector import path

    start = '2014/1/20 9:30'  # 示例起始日期
    instruments = ['SH000001', 'SH000002', 'SH000003', 'SH0000033']
    reader = CSVReader(instruments, start=start)
    reader.set_file_path(path=path)  # 定义类
    reader.load_csv_files()  # 加载数据
    registerdInstruments = reader.getRegisteredInstruments()  # 获取实际拥有的instruments,可能有些数据没有csv

    t = 0
    while not reader.eof() and t < 200:
        dateTime, df = reader.getNextValues()
        t += 1
        # print(dateTime, df)