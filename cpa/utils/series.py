from cpa.utils import collections
import numpy as np
import pandas as pd
from pyalgotrade import observer
from pyalgotrade import dataseries
DEFAULT_MAX_LEN = 1024


def get_checked_max_len(maxLen):
    if maxLen is None:
        maxLen = DEFAULT_MAX_LEN
    if not maxLen > 0:
        raise Exception("Invalid maximum length")
    return maxLen

class SequenceDataPanel:
    '''
    二维数据
    '''
    def __init__(self, colNames, maxLen=None, dtype=np.float32):
        '''
        :param maxLen: 默认按列增长,列为时间
        :param colNames: 行为codes
        '''

        self.__maxLen = get_checked_max_len(maxLen)
        self.__colLen = len(colNames)

        self.__cols = np.array(colNames)
        self.__dateTimes = collections.ListDeque(self.__maxLen) # rows
        self.__values = collections.NumpyPanelDeque(self.__maxLen, len(colNames), dtype=dtype)
        self.__newValuesEvent = observer.Event()

    def getMaxLen(self):
        return self.__maxLen

    def getNewValuesEvent(self):
        return self.__newValuesEvent

    def append(self, value):
        """Appends a value."""
        self.appendWithDateTime(None, value)

    def appendWithDateTime(self, dateTime, value):
        """
        Appends a value with an associated datetime.

        .. note::
            If dateTime is not None, it must be greater than the last one.
        """
        if dateTime is not None and len(self.__dateTimes) != 0 and self.__dateTimes[-1] >= dateTime:
            raise Exception("Invalid datetime. It must be bigger than that last one")

        assert(len(self.__values) == len(self.__dateTimes))
        self.__dateTimes.append(dateTime)
        self.__values.append(value)
        self.getNewValuesEvent().emit(self, dateTime, self.__values[-1])

    def to_frame(self):
        '''
        :return: 转换成datarame
        '''
        return pd.DataFrame(self.__values[:], index=pd.to_datetime(self.__dateTimes[:]), columns=self.__cols)

    def getDateTimes(self):
        return self.__dateTimes.data()

    def getColumnNames(self):
        '''
        :return: 返回列名
        '''
        return self.__cols

    def getRowSlice(self, rowName):
        '''
        :param rowName: 按行名返回Slice
        :return:
        '''
        return self.__values[np.where(self.__cols == rowName)[0][0], :]

    def getColSlice(self, colName):
        '''
        :param colName: 按列名返回slice
        :return:
        '''
        idx = np.where(self.__cols == colName)[0][0]
        return self.__values.getColumnsReference(idx)

    def __len__(self):
        return len(self.__values)

    def __contains__(self, key):

        return key in self.__cols

    def __getitem__(self, item):
        '''
        :param item: 默认按照行切索引
        :return:
        '''

        if isinstance(item, str):
            return self.getColSlice(colName=item)

        return self.__values[item]

    def data(self):
        return self.__values

    @property
    def shape(self):
        '''
        :return: 返回矩阵形态
        '''
        return (self.__maxLen, self.__colLen)

    def apply(self, func, axis=0):
        '''
        :param axis: 类似pandas的apply函数
        :return:
        '''
        if axis == 0:
            ret = np.empty(self.__colLen)
            for i in range(self.__colLen):
               ret[i] = func(self.__values[:, i])
        else:
            ret = np.empty(self.__maxLen)
            for i in range(self.__maxLen):
                ret[i] = func(self.__values[i, :])

        return ret



class SequenceDataSeries(dataseries.SequenceDataSeries):
    """A DataSeries that holds values in a sequence in memory.

    :param maxLen: The maximum number of values to hold.
        Once a bounded length is full, when new items are added, a corresponding number of items are discarded from the
        opposite end. If None then dataseries.DEFAULT_MAX_LEN is used.
    :type maxLen: int.
    """

    def __init__(self, maxLen=None):
        super(SequenceDataSeries, self).__init__(maxLen)

    def to_series(self, name=None):
        #change type from pyalgotrade dataseries to pandas Series
        return pd.Series(self.__values[:], index=self.getDateTimes(), name=name)

class SliceDataSeries:
    '''
    columns or row view of the 2-d numpy data panel,with index and names
    '''
    __slots__ = ('__superPanel', 'index', 'values', 'name', '__newValueEvent')

    def __init__(self, superPanel, indexValues, dataValues, name=None):

        self.__superPanel = superPanel
        self.index = indexValues
        self.values = dataValues
        self.name = name
        self.__superPanel.getNewValuesEvent().subscribe(self.onNewValue)
        self.__newValueEvent = observer.Event()

    def getMaxLen(self):
        return self.__superPanel.getMaxLen()

    def getSuperPanel(self):
        return self.__superPanel

    def getNewValueEvent(self):
        return self.__newValueEvent

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        return self.values[key]

    def onNewValue(self, dataPanel, dateTime, values):
        self.__newValueEvent.emit(self, dateTime, self.values[-1])

if __name__ == '__main__':

    a = SequenceDataPanel(['1','2','3'],3)
    a.appendWithDateTime('111',[1, 2, 3])
    b = a['2']
    print(b)
    print(b.data())
    a.appendWithDateTime('222',[2,3,4])
    print(b.data())
    print(a[:, :])
    print(a[:, -1][:])
    print(b[-1], b.data())


