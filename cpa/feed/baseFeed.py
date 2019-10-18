from collections import namedtuple#此处调用的是Python里自带的collections包
import collections

import numpy as np
from pyalgotrade import observer
from pyalgotrade import bar

from cpa.utils import series

Bar = namedtuple('Bar', 'dateTime open high low close volume')


class BarFeed:
    '''
    输入股票代码，时间序列，对应是开、高、低、收、量slice
    默认BarFeed不再注册事件，使用主动调用的方式处理，单个Series具有事件，在指标更新时使用,
    **但不建议使用该方法，尽量使用panelIndicator 或者 dataPanel.apply处理**
    单个instrument 对应的OHLC 数据切片
    noneNaClose 为非切片
    '''
    __slots__ = (
        '__dataFeed',  # 所切片Feed
        '__instrument',  # 代码
        '__dateTimes',  # 时间序列
        '__openSlice',  # 开盘价
        '__highSlice',  # 最高价
        '__lowSlice',  # 最低价
        '__closeSlice',  # 收盘价
        '__volumeSlice',  # 交易量
        '__noneNaClose',
        '__newCloseEvent',
        '__newNoneNaCloseEvent',
        '__frequency',
        '__maxLen',
    )

    def __init__(self, dataFeed, instrument, dateTimes, _open, _high, _low, _close, volume, frequency, maxLen=None):
        """slice 长度恒定, 须按照现有数据行长度剪切, sliceSeries保留事件驱动"""

        self.__dataFeed = dataFeed
        self.__instrument = instrument
        self.__dateTimes = dateTimes
        self.__openSlice = _open
        self.__highSlice = _high
        self.__lowSlice = _low
        self.__closeSlice = _close
        self.__volumeSlice = volume
        self.__frequency = frequency
        self.__maxLen = maxLen

    def __len__(self):  # 获取时间长度
        return self.__dateTimes.__len__()

    def getInstrument(self):  # 获取股票代码
        return self.__instrument

    def getFrequency(self):
        return self.__frequency

    def getMaxLen(self):
        return self.__maxLen

    def getOpenSlice(self):
        return self.__openSlice

    def getHighSlice(self):
        return self.__highSlice

    def getLowSlice(self):
        return self.__lowSlice

    def getCloseSlice(self):
        return self.__closeSlice

    def getVolumeSlice(self):
        return self.__volumeSlice

    def getOpenSeries(self):
        '''
        :return: 返回带索引的sliceSeries，索引为dateTime
        '''
        return series.SliceDataSeries(self.__dataFeed.openPanel, self.__dateTimes, self.__openSlice, name='open')

    def getHighSeries(self):
        return series.SliceDataSeries(self.__dataFeed.highPanel, self.__dateTimes, self.__highSlice, name='high')

    def getLowSeries(self):
        return series.SliceDataSeries(self.__dataFeed.lowPanel, self.__dateTimes, self.__lowSlice, name='low')

    def getCloseSeries(self):
        return series.SliceDataSeries(self.__dataFeed.closePanel, self.__dateTimes, self.__closeSlice, name='close')

    def getVolumeSeries(self):
        return series.SliceDataSeries(self.__dataFeed.volumePanel, self.__dateTimes, self.__volumeSlice, name='volume')

    def getDateTimes(self):  # 获取所有时间序列
        return self.__dateTimes

    def getLastBar(self):  # 获取最后一个时间的值
        return Bar(dateTime=self.__dateTimes[-1],
                   open=self.__openSlice[-1],
                   high=self.__highSlice[-1],
                   low=self.__lowSlice[-1],
                   close=self.__closeSlice[-1],
                   volume=self.__volumeSlice[-1])


class PanelFeed:


    # 事件优先级, 重采样要高于计算因子
    class EventPriority:
        PREFILTER = 1000
        INDICATOR = 2000
        RESAMPLE = 3000
        FACTOR = 4000

        @classmethod
        def getEventsType(cls):
            return sorted([1000, 2000, 3000, 4000])

    def __init__(self, dataSource, instruments, frequency=bar.Frequency.MINUTE, maxLen=None):

        self.dataSource = dataSource
        self.instruments = instruments
        self.maxLen = maxLen
        self.frequency = frequency
        self.__currentDatetime = np.nan

        self.openPanel = series.SequenceDataPanel(instruments, self.maxLen, dtype=np.float32)  # 定义了一个类
        self.highPanel = series.SequenceDataPanel(instruments, self.maxLen, dtype=np.float32)
        self.lowPanel = series.SequenceDataPanel(instruments, self.maxLen, dtype=np.float32)
        self.closePanel = series.SequenceDataPanel(instruments, self.maxLen, dtype=np.float32)
        self.volumePanel = series.SequenceDataPanel(instruments, self.maxLen)

        self.__panelEvents = collections.OrderedDict({e: observer.Event() for e in self.EventPriority.getEventsType()})

        self.barFeeds = {}
        for instrument in instruments:
            self.barFeeds[instrument] = BarFeed(self,
                                                instrument=instrument,
                                                dateTimes=self.closePanel.getDateTimes(),
                                                _open=self.openPanel[instrument],
                                                _high=self.highPanel[instrument],
                                                _low=self.lowPanel[instrument],
                                                _close=self.closePanel[instrument],
                                                volume=self.volumePanel[instrument],
                                                frequency=self.frequency,
                                                maxLen=self.maxLen)
    def getCurrentDatetime(self):
        return self.__currentDatetime

    def getInstruments(self):
        return self.instruments

    def getFrequency(self):
        return self.frequency

    def getNewPanelsEvent(self, priority=None):

        assert priority in self.EventPriority.getEventsType()
        return self.__panelEvents[priority]

    def eof(self):
        return self.dataSource.eof()

    def getOpenPanel(self):
        return self.openPanel

    def getHighPanel(self):
        return self.highPanel

    def getLowPanel(self):
        return self.lowPanel

    def getClosePanel(self):
        return self.closePanel

    def getVolumePanel(self):
        return self.volumePanel

    def dispatchNewValueEvent(self, *args, **kwargs):

        for key, evt in self.__panelEvents.items():
            evt.emit(*args, **kwargs)

    def getNextValues(self):

        dateTime, df = self.dataSource.getNextValues()
        self.__currentDatetime = dateTime

        self.openPanel.appendWithDateTime(dateTime, df['open'].sort_index().values)
        self.highPanel.appendWithDateTime(dateTime, df['high'].sort_index().values)
        self.lowPanel.appendWithDateTime(dateTime, df['low'].sort_index().values)
        self.closePanel.appendWithDateTime(dateTime, df['close'].sort_index().values)
        self.volumePanel.appendWithDateTime(dateTime, df['volume'].sort_index().values)

        self.dispatchNewValueEvent(self, dateTime, df)

        return dateTime, df

    def run(self, stopCount=None):
        counter = 0
        while not self.eof():
             counter += 1
             self.getNextValues()
             if stopCount is not None and counter > stopCount:
                 break

if __name__ == '__main__':

    import os
    from cpa.utils.pathSelector import path
    from cpa.dataReader.csvReader import CSVReader
    path = 'G:/data/data2'  # 个股数据

    start = '2015/1/20 9:30'  # 示例起始日期
    # instruments = ['SH600233', 'SH600236', 'SZ000005']
    instruments = os.listdir(path)  # 读取文件夹下所有文件名
    instruments = [file.split('.')[0] for file in instruments]  # instruments必须要足够多，后续需分成20组
    reader = CSVReader(instruments, start=start)
    reader.set_file_path(path=path)  # 定义类
    reader.load_csv_files()  # 加载数据
    registerdInstruments = reader.getRegisteredInstruments()  # 获取实际拥有的instruments,可能有些数据没有csv
    panelFeed = PanelFeed(reader, registerdInstruments, maxLen=1024)

    t = 0
    while not reader.eof() and t < 1024:

        dateTime, df = panelFeed.getNextValues()
        t += 1
        print(dateTime, df)

    panelFeed.barFeeds['SH600236'].getLastBar()
    panelFeed.barFeeds['SH600233']

