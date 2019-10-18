# -*- coding: utf-8 -*-
'''
@Time    : 2019/6/10 22:03
@Author  : msi
@Email   : sdu.xuefu@gmail.com
'''

from cpa.factorPool import BaseBarFeedCalculator
from cpa.indicators.seriesIndicators import ma

class Ma(BaseBarFeedCalculator):
    '''
    以ma作为因子值的示例
    '''

    def __init__(self, factorManager, barFeed):
        super().__init__(factorManager, barFeed)
        self.factorManager = factorManager
        self.closeSeries = barFeed.getCloseSeries()
        self.ma_20 = ma.MA(self.closeSeries, 20, self.closeSeries.getMaxLen())

    def calScore(self, barFeed, dateTime, bar):
        return self.closeSeries[-10:].mean() / self.ma_20[-1]


if __name__ == "__main__":
    from cpa.factorPool import factorBase
    from cpa.dataReader import csvReader
    from cpa.feed import baseFeed
    from cpa.utils.pathSelector import path

    start = '2014/5/1 9:30'  # 示例起始日期
    instruments = ['SH000001', 'SH000002', 'SH000003', 'SH000004', 'SH000005']
    reader = csvReader.CSVReader(instruments, start=start)
    reader.set_file_path(path=path)  # 定义类
    reader.load_csv_files()  # 加载数据

    registerdInstruments = reader.getRegisteredInstruments()
    panelFeed = baseFeed.PanelFeed(reader, registerdInstruments, maxLen=1024)
    rawFactor = factorBase.BaseFactorManager(panelFeed, Ma, 1024)  # series形式

    i = 0
    while not panelFeed.eof() and i < 1000:
        i += 1
        dateTime, df = panelFeed.getNextValues()

    # 数据展示
    print(rawFactor.to_frame())
