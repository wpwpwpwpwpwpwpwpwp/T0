from cpa.utils.series import SequenceDataSeries


class MA(SequenceDataSeries):
    '''
    **不可对sliceSeries赋值,若赋值须copy一份**
     不推荐使用seriesIndicator, 尽量使用panelIndicator或者
    '''

    def __init__(self, sliceSeries, n, maxLen=None):
        super(MA, self).__init__(maxLen=maxLen)  # 继承的子类
        sliceSeries.getNewValueEvent().subscribe(self.onNewValue)
        self.n = n

    def onNewValue(self, sliceSeries, dateTime, value):
        '''
        :return:最新的一行值
        '''
        value = sliceSeries[-self.n: ].mean(axis=0)  # 取最新一期值
        self.appendWithDateTime(dateTime, value)


if __name__ == '__main__':
    from cpa.dataReader import csvReader
    from cpa.utils.pathSelector import path
    from cpa.feed import baseFeed

    start = '2014/5/1 9:30'  # 示例起始日期
    instruments = ['SH000001', 'SH000002', 'SH000003', 'SH000004', 'SH000005']

    reader = csvReader.CSVReader(instruments, start=start)
    reader.set_file_path(path=path)  # 定义类
    reader.load_csv_files()  # 加载数据

    registerdInstruments = reader.getRegisteredInstruments()
    panelFeed = baseFeed.PanelFeed(reader, registerdInstruments, maxLen=1024)  #
    maClass = MA(panelFeed.barFeeds[registerdInstruments[0]].getCloseSeries(), n=20, maxLen=1024)

    t = 0
    while not panelFeed.eof() and t < 1100:
        dateTime, df = panelFeed.getNextValues()  # 更新基础数据
        print(maClass[-1])
        t += 1
