'''
基础的因子管理模块，管理因子计算类，处理得到的打分值，并更新因子计算矩阵
@Time    : 2019/6/10 21:24
@Author  : msi
@Email   : sdu.xuefu@gmail.com
'''
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from cpa import factorPool
from cpa.feed import baseFeed
from cpa.utils.series import SequenceDataPanel


class FactorPanel(SequenceDataPanel):

    def __init__(self, panelFeed, factorCalculatorCls, maxLen=None):
        '''
        :param panelFeed:原始的数据feed
        :param factorCalculatorCls: 因子计算类
        :param maxLen: 存储因子值的最大长度
        :indicator 如果需要内置指标从此处初始化
        :return:
        '''
        super().__init__(colNames=panelFeed.getInstruments(), maxLen=maxLen)

        self.panelFeed = panelFeed
        self.instruments = panelFeed.getInstruments()
        self.colLen = len(self.instruments)

        self.factorName = factorCalculatorCls.name
        self.initializeFactorCals(factorCalculatorCls)

        # 注册回调事件，优先级为因子计算
        self.panelFeed.getNewPanelsEvent(priority=baseFeed.PanelFeed.EventPriority.FACTOR).subscribe(self.onNewValues)

    def initializeFactorCals(self, factorCalculatorCls):
        '''
        :return: 初始化打分结果存储和计算对象
        1.如果使用barFeed进行打分,需要挨个生成计算器对象,并存储在barFeedScoreCals字典中
        2.使用panelFeed直接打分,只需要生成一个并存储在panelFeedScoreCals字典中
        '''

        self.barFeedFactorCal, self.panelFeedFactorCal = None, None

        if issubclass(factorCalculatorCls, factorPool.BaseBarFeedCalculator):
            factorCalculators = []
            for instrument in self.panelFeed.barFeeds:
                factorCalculator = factorCalculatorCls(self, self.panelFeed.barFeeds[instrument])
                factorCalculators.append(factorCalculator)

            self.barFeedFactorCal = factorCalculators

        elif issubclass(factorCalculatorCls, factorPool.BasePanelCalculator):
            factorCalculator = factorCalculatorCls(self, self.panelFeed)
            self.panelFeedFactorCal = factorCalculator

    def calRawScores(self, panelFeed, dateTime, df):
        '''
        :param panelFeed:
        :param dateTime:
        :return:
        '''
        # barFeed类因子计算
        if self.barFeedFactorCal is not None:
            score = np.zeros((self.colLen,))
            for j, instrument in enumerate(self.instruments):
                sliceFeed = self.panelFeed.barFeeds[instrument]
                score[j] = self.barFeedFactorCal[j].calScore(sliceFeed, dateTime, sliceFeed.getLastBar())

        # panelFeed类因子计算
        else:
            score = self.panelFeedFactorCal.calScore(panelFeed, dateTime, df)

        return score


    def onNewValues(self, panelFeed, dateTime, df):
        rawScore = self.calRawScores(panelFeed, dateTime, df)
        self.appendWithDateTime(dateTime, rawScore)

    def factorPlot(self, codes, seriesName):
        '''
        :param codes:可选作哪个或哪组code的图
        :param seriesName: OHLC以及score作图
        :return:
        '''
        if seriesName == 'score':
            plotData = pd.DataFrame(index=self.getDateTimes(), columns=self.getColumnNames(), data=self[:, :])
        elif seriesName == 'open':
            self.priceData = self.panelFeed.openPanel
        elif seriesName == 'close':
            self.priceData = self.panelFeed.closePanel
        elif seriesName == 'high':
            self.priceData = self.panelFeed.highPanel
        elif seriesName == 'low':
            self.priceData = self.panelFeed.lowPanel

            plotData = pd.DataFrame(index=self.priceData.getDateTimes(), columns=self.priceData.getColumnNames(),
                                    data=self.priceData[:, :])
            plotData = plotData.dropna(axis=1, how='any')  # 去除含nan的股票数据
            plotData = plotData / plotData.iloc[0, :]  # 消除量纲的影响，起始点均为1

        thisPlotDataData = plotData[codes]
        timeList = list(thisPlotDataData.index)
        N = len(thisPlotDataData)
        ind = np.arange(N)
        fig, ax = plt.subplots(1, 1)
        ax.plot(ind, thisPlotDataData)
        plt.legend(list(thisPlotDataData.columns), loc='best')

        def format_date(x, pos=None):  # 改变横坐标格式
            if x < 0 or x > len(timeList) - 1:
                return ''
            else:
                return timeList[int(x)]

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        fig.show()

    def getFactorCalculator(self, instrument=None):
        '''
        :param instrument: 返回因子计算器对象，如果是单个因子计算的返回数组或者依据instrument返回该计算对象
                            面板计算则直接返回面板计算器
        :return:
        '''
        assert (self.panelFeedFactorCal and self.barFeedFactorCal) is None # 限定只有一个计算器
        if self.barFeedFactorCal:
            return self.barFeedFactorCal if self.instruments is None else self.barFeedFactorCal[np.where(self.instruments == instrument)[0][0]]
        else:
            return self.panelFeedFactorCal


if __name__ == '__main__':
    from cpa.factorPool.example import maPanelFactor
    from cpa.dataReader import csvReader
    from cpa.utils.pathSelector import path

    start = '2014/5/1 9:30'  # 示例起始日期
    instruments = ['SH000001', 'SH000002', 'SH000003', 'SH000004', 'SH000005']

    reader = csvReader.CSVReader(instruments, start=start)
    reader.set_file_path(path=path)  # 定义类
    reader.load_csv_files()  # 加载数据

    registerdInstruments = reader.getRegisteredInstruments()
    panelFeed = baseFeed.PanelFeed(reader, registerdInstruments, maxLen=1024)

    rawFactor = FactorPanel(panelFeed, maPanelFactor.Ma, 1024)  # panel形式

    i = 0
    while not panelFeed.eof() and i < 1100:
        i += 1
        dateTime, df = panelFeed.getNextValues()
    # 数据展示

    print(rawFactor.to_frame())
    # 作图
    codes = ['SH000001', 'SH000003', 'SH000004']
    rawFactor.factorPlot(codes, seriesName='open')  # 起始值均为1
    rawFactor.factorPlot(codes, seriesName='score')