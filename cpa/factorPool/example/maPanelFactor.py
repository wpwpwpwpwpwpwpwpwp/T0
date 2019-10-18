import numpy as np

from cpa import factorPool
from cpa.indicators.panelIndicators import ma


class Ma(factorPool.BasePanelCalculator):
    '''
    以ma作为因子值的示例，同时添加指标
    '''

    def __init__(self, factorManager, panelFeed):
        super().__init__(factorManager, panelFeed)
        self.factorManager = factorManager

        self.ma_10 = ma.MA(panelFeed.closePanel, 10, panelFeed.maxLen)
        self.ma_20 = ma.MA(panelFeed.closePanel, 20, panelFeed.maxLen)

    def calScore(self, barFeed, dateTime, df):
        return np.divide(self.ma_10[-1, :], self.ma_20[-1, :])


if __name__ == '__main__':
    from cpa.dataReader import csvReader
    from cpa.utils.pathSelector import path
    from cpa.factorPool import factorBase
    import os
    from cpa.indicators.panelIndicators import returns
    from cpa.feed import baseFeed
    from cpa.factorProcessor.factorTest import DefaultFactorTest

    start = '2014/5/1 9:30'  # 示例起始日期
    instruments = os.listdir(path)  # 读取文件夹下所有文件名
    instruments = [file.split('.')[0] for file in instruments]  # instruments必须要足够多，后续需分成20组

    reader = csvReader.CSVReader(instruments, start=start)
    reader.set_file_path(path=path)  # 定义类
    reader.load_csv_files()  # 加载数据

    registerdInstruments = reader.getRegisteredInstruments()
    panelFeed = baseFeed.PanelFeed(reader, registerdInstruments, maxLen=1024)
    rawFactor = factorBase.BaseFactorManager(panelFeed, dma, 1024)  # panel形式
    F = 30  # 调仓频率
    _return = returns.Returns(panelFeed, n=F, maxLen=1024)  # 以开盘价计算的向前n期收益
    factorTester = DefaultFactorTest(panelFeed, rawFactor, _return,
                                     indicators=['IC', 'rankIC', 'beta', 'gpIC', 'tbdf'], maxLen=2048,
                                     lag=F,
                                     ngroup=20, cut=0.1)  # 定义因子评价类

    t = 0
    while not panelFeed.eof() and t < 4000:
        dateTime, df = panelFeed.getNextValues()
        t += 1

    # 数据展示
    print(factorTester.ICseries.to_series('ic'))

    # # 作图
    factorTester.plotAll()
