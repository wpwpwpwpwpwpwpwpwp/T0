# -*- coding: utf-8 -*-
import numpy as np

from cpa.utils.series import SequenceDataPanel


class Returns(SequenceDataPanel):
    '''
    **不可对panel赋值,若赋值须copy一份**
    收益率计算
    '''

    def __init__(self, panelFeed, n, maxLen):
        super().__init__(panelFeed.closePanel.getColumnNames(), maxLen=maxLen)
        panelFeed.getNewPanelsEvent(priority=panelFeed.EventPriority.INDICATOR).subscribe(self.onNewValues)
        self.n = n

    def onNewValues(self, panelFeed, dateTime, df):
        '''
        :param ReturnValue: 存储数据
        :return:最新的一行值
        '''
        panelData = panelFeed.getClosePanel()
        if panelData.__len__() < self.n + 1:
            self.appendWithDateTime(dateTime, np.full(panelData.shape[1], np.nan))  # 若数据量不足，返回nan值
        else:
            _return = panelData[-1, :] / panelData[-self.n - 1, :] - 1
            self.appendWithDateTime(dateTime, _return)
        return self


if __name__ == '__main__':
    from cpa.dataReader import csvReader
    from cpa.utils.pathSelector import path
    from cpa.feed import baseFeed
    from cpa.indicators.panelIndicators import returns

    start = '2014/5/1 9:30'  # 示例起始日期
    instruments = ['SH000001', 'SH000002', 'SH000003', 'SH000004', 'SH000005']

    reader = csvReader.CSVReader(instruments, start=start)
    reader.set_file_path(path=path)  # 定义类
    reader.load_csv_files()  # 加载数据

    registerdInstruments = reader.getRegisteredInstruments()
    panelFeed = baseFeed.PanelFeed(reader, registerdInstruments, maxLen=1024)  # 定义取数panel类
    _returns = returns.Returns(panelFeed, n=30, maxLen=1024)  # 以开盘价计算的向前n期收益,定义returns类

    t = 0
    while not panelFeed.eof() and t < 1100:
        dateTime, df = panelFeed.getNextValues()  # 更新基础数据
        t += 1

    print(_returns.to_frame())