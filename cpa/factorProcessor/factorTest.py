"""
因子检验模块
"""
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from cpa.calculator import sectionCalculator
from cpa.feed import baseFeed
from cpa.utils import plotter
from cpa.utils import series
warnings.filterwarnings('ignore')
class DefaultFactorTest:
    """
    默认因子检验模块
    """

    def __init__(self, panelFeed, factorPanel, returnPanel, indicators, lag, ngroup, cut, maxLen):
        '''
        :param panelFeed:
        :param factorPanel:
        :param returnPanel:
        :param indicators: 需要计算的指标List，indicators = ['IC', 'rankIC', 'beta', 'gpIC', 'tbdf']
        :param maxLen:
        :param lag: 收益序列取lag, lag至少为1, 即取上一个score同当前收益取截面, 前lag期因子值对应当期收益
        :param ngroup: 分成ngroup组，分ngroup组后的相关系数
        :param cut: 分组信息，0.1代表前10 % -后10 %
        '''
        self.panelFeed = panelFeed
        self.factorPanel = factorPanel
        self.returnPanel = returnPanel
        self.indicators = indicators
        self.lag = lag
        self.ngroup = ngroup
        self.cut = cut
        self.maxLen = maxLen

        self.ICseries = series.SequenceDataSeries(self.maxLen)
        self.rankICseries = series.SequenceDataSeries(self.maxLen)
        self.betaSeries = series.SequenceDataSeries(self.maxLen)
        self.gpICSeries = series.SequenceDataSeries(self.maxLen)
        self.tbdfSeries = series.SequenceDataSeries(self.maxLen)  # top组平均收益-bottom组平均收益
        # 注册回调事件，优先级为因子计算
        self.panelFeed.getNewPanelsEvent(priority=baseFeed.PanelFeed.EventPriority.FACTOR).subscribe(
            self.updateIndicators)
    def updateIndicators(self, panelFeed, dateTime, df):
        '''
        :param scorePanel: 分值矩阵
        :param returnPanel: 收益矩阵
        :return:因子效果评价指标，IC,rankIC,beta,gpIC,tbdf
        '''
        if len(self.returnPanel) > 2 * self.lag:  # 前期数据过短时，return值不完善，比return向前lag期因子值也不完善，因此定为2 * self.lag
            self.dateTime = dateTime
            thisAllReturn = self.returnPanel[-1, :]  # 当期所有数据
            lastAllFactor = self.factorPanel[-self.lag, :]  # 向前lag期因子
            returnNotNan = np.argwhere(1 - np.isnan(thisAllReturn))  # 找出非nan值的位置
            factorNotNan = np.argwhere(1 - np.isnan(lastAllFactor))
            # 股票因子数量小于2无法分层
            if factorNotNan.__len__() < 1:
                return
            notNanLocate = np.intersect1d(returnNotNan, factorNotNan)  # 求两者交集
            self.thisReturn = np.nan_to_num(thisAllReturn[notNanLocate].reshape((len(notNanLocate),)))  # 当期去除nan后的数据
            self.lastFactor = np.nan_to_num(lastAllFactor[notNanLocate].reshape((len(notNanLocate),)))

            if 'IC' in self.indicators:
                IC = sectionCalculator.IC(self.lastFactor, self.thisReturn)
                if 1 - np.isnan(IC):
                    self.ICseries.appendWithDateTime(self.dateTime, IC)  # Pearson 相关系数

            if 'rankIC' in self.indicators:
                rankIC = sectionCalculator.RKIC(self.lastFactor, self.thisReturn)
                if 1 - np.isnan(rankIC):
                    self.rankICseries.appendWithDateTime(self.dateTime, rankIC)  # Spearman 相关系数

            if 'gpIC' in self.indicators:
                gpIC = sectionCalculator.GPIC(self.lastFactor, self.thisReturn, self.ngroup)
                if 1 - np.isnan(gpIC):
                    self.gpICSeries.appendWithDateTime(self.dateTime, gpIC)  # 分n组后的相关系数

            if 'beta' in self.indicators:
                beta = sectionCalculator.BETA(self.lastFactor, self.thisReturn)
                if 1 - np.isnan(beta):
                    self.betaSeries.appendWithDateTime(self.dateTime, beta)  # 单因子回归斜率

            if 'tbdf' in self.indicators:
                tbdf = sectionCalculator.TBDF(self.lastFactor, self.thisReturn, self.cut)
                if 1 - np.isnan(tbdf):
                    self.tbdfSeries.appendWithDateTime(self.dateTime, tbdf)  # top平均收益 - bottom平均收益
    def plotAll(self):
        ICseries = self.ICseries.to_series('IC')
        rankICseries = self.rankICseries.to_series('rankIC')
        gpICSeries = self.gpICSeries.to_series('gpIC')
        betaSeries = self.betaSeries.to_series('beta')
        tbdfSeries = self.tbdfSeries.to_series('tbdf')
        data = pd.concat([ICseries, rankICseries, betaSeries, gpICSeries, tbdfSeries], axis=1)
        idx = pd.to_datetime(data.index)
        data = data.reset_index()
        fig, ax = plt.subplots(3, 1, sharex='col')
        ax[0].set_title('IC_CUMSUM')
        ax[0].plot(data.IC.cumsum(), label='IC')
        ax[0].plot(data.rankIC.cumsum(), label='rankIC')
        ax[0].plot(data.gpIC.cumsum(), label='gpIC')
        ax[0].legend(loc=3)
        ax[1].set_title('BETA')
        ax[1].plot(data.beta, label='beta')
        ax[1].legend(loc=3)
        ax[2].set_title('TBDF_CUMSUM')
        ax[2].plot(data.tbdf.cumsum(), label='tbdf')
        ax[2].legend(loc=3)
        ax[2].xaxis.set_major_formatter(plotter.MyFormatter(idx, fmt='%Y-%m-%d %H:%M'))
        plt.tight_layout()
        plt.show()
    def plotSumCurve(self, testSeries):
        testData = testSeries[:]
        allTimeList = testSeries.getDateTimes()
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.array(testData).cumsum(), label='cumsum')
        plt.legend(loc='best')
        def format_date(x, pos=None):  # 改变横坐标格式
            if x < 0 or x > len(allTimeList) - 1:
                return ''
            else:
                return allTimeList[int(x)]
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))  # 将横坐标设置为日期
        fig.show()
    def plotProdCurve(self, testSeries):
        testData = testSeries[:]
        allTimeList = testSeries.getDateTimes()
        testData = 1 + testData
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.array(testData).cumprod(), label='prod')
        plt.legend(loc='best')
        def format_date(x, pos=None):  # 改变横坐标格式
            if x < 0 or x > len(allTimeList) - 1:
                return ''
            else:
                return allTimeList[int(x)]
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        fig.show()
if __name__ == '__main__':
    from cpa.factorPool.example import maPanelFactor
    from cpa.dataReader import csvReader
    from cpa.utils.pathSelector import path
    from cpa.factorPool import factorBase
    import os
    from cpa.indicators.panelIndicators import returns
    start = '2014/5/1 9:30'  # 示例起始日期
    instruments = os.listdir(path)  # 读取文件夹下所有文件名
    instruments = [file.split('.')[0] for file in instruments]  # instruments必须要足够多，后续需分成20组

    reader = csvReader.CSVReader(instruments, start=start)
    reader.set_file_path(path=path)  # 定义类
    reader.load_csv_files()  # 加载数据

    registerdInstruments = reader.getRegisteredInstruments()
    panelFeed = baseFeed.PanelFeed(reader, registerdInstruments, maxLen=1024)
    rawFactor = factorBase.FactorPanel(panelFeed, maPanelFactor.Ma, 1024)  # panel形式
    F = 240  # 调仓频率
    _return = returns.Returns(panelFeed, n=F, maxLen=1024)  # 以开盘价计算的向前n期收益
    factorTester = DefaultFactorTest(panelFeed, rawFactor, _return,
                                     indicators=['IC', 'rankIC', 'beta', 'gpIC', 'tbdf'], maxLen=2048,
                                     lag=F,
                                     ngroup=20, cut=0.4)  # 定义因子评价类

    t = 0
    while not panelFeed.eof() and t < 1100:
        dateTime, df = panelFeed.getNextValues()
        t += 1

    # 数据展示
    print(factorTester.ICseries.to_series('ic'))

    # # 作图
    factorTester.plotAll()
