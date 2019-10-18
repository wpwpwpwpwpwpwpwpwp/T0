from cpa.utils.series import SequenceDataPanel


class MA(SequenceDataPanel):
    '''
    **不可对panel赋值,若赋值须copy一份**
    panel行为时间,列为codes,值为数据矩阵  
    '''

    def __init__(self, dataPanel, n, maxLen):
        super(MA, self).__init__(dataPanel.getColumnNames(), maxLen=maxLen)  # 继承的子类
        dataPanel.getNewValuesEvent().subscribe(self.onNewValues)
        self.n = n

    def onNewValues(self, dataPanel, dateTime, values):
        '''
        :return:最新的一行值
        '''

        values = dataPanel[-self.n:, :].mean(axis=0)  # 取最新一期值
        self.appendWithDateTime(dateTime, values)


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
    panelFeed = baseFeed.PanelFeed(reader, registerdInstruments, maxLen=1024)  # 定义取数panel类
    maPanel = MA(panelFeed.closePanel, n=20, maxLen=1024)  # 以开盘价计算的向前n期收益,定义returns类

    t = 0
    while not panelFeed.eof() and t < 1100:
        dateTime, df = panelFeed.getNextValues()  # 更新基础数据
        t += 1

    # 数据展示
    print(maPanel.to_frame())