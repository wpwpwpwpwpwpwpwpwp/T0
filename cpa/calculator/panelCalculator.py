'''
cal by panel, numpy 2-d array
'''
# coding=utf8
__author__ = 'wangjp'
import numpy as np
import pandas as pd
class CalculatorPanel:
    """
        基础计算函数 ：
        1.传入SequenceDataPanel，以numpy为底层，numpy的函数都可以用
        2.计算结果为最新一行，raw，每驱动一次计算一行

        1. 矩阵版 计算函数库, 主要目标 通过减少 groupby 步骤加速计算
        2. rowIndex 时间 colIndex stkcd
        3.对于数据中的缺失值，基本的处理思想是尽量保持原有数据提供的信息，如 计算五日均值
          数据有缺失 [1,2,nan,nan,3] 则使用其中有的三天数据进行计算
    """
    def __init__(self):
        pass
    @classmethod
    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    @classmethod
    def power(cls, x, n):
        return np.sign(x) * np.power(x.abs(), n)
    @classmethod
    def cmpMin(cls, x, y):
        return x * (x <= y) + y * (y < x)
    @classmethod
    def cmpMax(cls, x, y):
        return x * (x >= y) + y * (y > x)
    @classmethod
    def Delay(cls, x, num, fillNan=None):
        raw = x.shift(periods=num)
        if fillNan is not None:
            raw[raw.isnull()] = fillNan
        return raw
    @classmethod
    def Diff(cls, x, num, fillNan=None):
        raw = x - CalculatorPanel.Delay(x=x, num=num, fillNan=fillNan)
        return raw
    @classmethod
    def Max(cls, x, num, minobs=0):
        raw=CalculatorPanel.rolling_window(x,num)
        raw = raw.max()
        raw = raw.sort_index()
        return raw
    @classmethod
    def Min(cls, x, num,  minobs=0):
        raw = CalculatorPanel.rolling_window(x, num)
        raw = raw.min()
        raw = raw.sort_index()
        return raw

    @classmethod
    def Median(cls, x, num, minobs=0):
        raw = CalculatorPanel.rolling_window(x, num)
        raw = raw.median()
        raw = raw.sort_index()
        return raw
    @classmethod
    def Mean(cls, x, num, minobs=0):
        """
        窗口期内的 NaN 数量满足 minNum 将被忽略
        :param x:
        :param num:
        :param by:
        :return:
        """
        raw = x[-num:, :].mean(axis=0)
        return raw
    @classmethod
    def Sum(cls, x, num, minobs=0):
        raw = num * CalculatorPanel.Mean(x=x, num=num, minobs=minobs)
        return raw
    @classmethod
    def Std(cls, x, num, ddof=1, minobs=0):
        raw = x[-num:, :].std(axis=0)
        return raw
    @classmethod
    def Var(cls, x, num, ddof=1, minobs=0):
        raw = x[-num:, :].var(axis=0)
        return raw
    @classmethod
    def Skew(cls, x, num, minobs=0):
        raw=CalculatorPanel.rolling_window(x,num)
        raw = raw.skew()
        raw = raw.sort_index()
        return raw
    @classmethod
    def Kurt(cls, x, num, minobs=0):
        raw = CalculatorPanel.rolling_window(x, num)
        raw = raw.kurt()
        raw = raw.sort_index()
        return raw
    @classmethod
    def Countif(cls, condition, num, minobs=0):
        condition[condition.isnull()] = 0
        raw = CalculatorPanel.Sum(x=condition, num=num, minobs=minobs)
        return raw
    @classmethod
    def Sumif(cls, x, condition, num, minobs=0):
        raw = CalculatorPanel.rolling_window(x*condition, num)
        raw = raw.sum()
        raw = raw.sort_index()
        return raw
    @classmethod
    def Sma(cls, x, n, m):
        # Xt = At*m/n + Xt-1*(n-m)/n  ie . alpha = m/n com = n/m - 1
        assert n>m
        def ewm(val):
            return val.ewm(com=com).mean()
        com = n/m - 1
        raw = x.apply(ewm, axis=0)
        return raw
    @classmethod
    def Wma(cls, x, num, pct=0.9, weightType='exp', minobs=0):
        """
        计算前 num期样本加权平均值
        if all nan, then 0/0 makes nan
        :param x:
        :param num:
        :param pct:
        :param weightType:  权重方式
        :param minobs:
        :return:
        """
        if weightType == 'exp':      # 通过指数等比级数加权
            weights = (pct ** np.array(range(num-1, -1, -1))).reshape(-1, 1)
            totWeight = (1 - pct**num) / (1-pct)
        elif weightType == 'halflife':      # 通过半衰期加权， pct 表示半衰期 单位 年
            rate = -np.log(2) / pct
            weights = (np.exp(rate * np.array(range(num-1, -1, -1)))).reshape(-1, 1)
            totWeight = np.sum(weights)
        else:
            raise NotImplementedError
        xValues = x.values
        nanIdx = np.isnan(x)
        xValues[nanIdx] = 0
        noNa = (~nanIdx).astype(np.int).values
        raw = np.full(x.shape, np.nan)
        adjfct = np.full(x.shape, np.nan)
        for dumi in range(minobs, x.shape[0]):
            head = max(dumi - num + 1, 0)
            tail = dumi + 1     # 确保当前值也被切割
            count = tail - head
            raw[dumi, :] = np.sum(xValues[head:tail, :] * weights[:count], axis=0)
            adjfct[dumi, :] = np.sum(noNa[head:tail, :] * weights[:count], axis=0)
        raw = totWeight / adjfct * raw      # nan 对应设为0， 导致求和后数量级偏小，通过 totWeigt / adjfct 调整
        raw[nanIdx] = np.nan  # 保留原始数据中的NaN
        raw = pd.DataFrame(raw, index=x.index, columns=x.columns)
        return raw
    @classmethod
    def Decaylinear(cls, x, d):
        # 对 A 序列计算移动平均加权，其中权重对应 d,d-1,…,1/ sum(1-d)（权重和为 1）
        # if all nan, then 0/0 makes nan
        xValues = x.values
        nanIdx = np.isnan(x)
        xValues[nanIdx] = 0
        noNa = (~nanIdx).astype(np.int).values
        raw = 0
        adjfct = 0
        for dumi in range(d, 0, -1):
            tail = d - dumi
            xShift = np.roll(xValues, shift=tail, axis=0)
            adjShift = np.roll(noNa, shift=tail, axis=0)
            xShift[:tail, :] = 0
            adjShift[:tail, :] = 0
            raw += dumi * xShift
            adjfct += dumi * adjShift
        adjfct = adjfct.astype(np.float)
        adjfct[adjfct == 0] = np.nan
        raw = raw / adjfct
        raw[nanIdx] = np.nan  # 保留原始数据中的NaN
        raw = pd.DataFrame(raw, index=x.index, columns=x.columns)
        return raw

    @classmethod
    def TsToMin(cls, x, num, minobs=0):
        """
        计算 当前值 距离窗口期内最小值之间的间隔数, 当前值本身也算一位
        :param x:
        :param num:
        :param minobs:
        :return:
        """
        xValues = x.values
        nanIdx = np.isnan(xValues)
        xValues[nanIdx] = 0     # 为正常调用 nanargmax, 把空值设为0
        raw = np.full(x.shape, np.nan)
        for dumi in range(minobs, x.shape[0]):
            head = max(dumi - num + 1, 0)
            tail = dumi + 1     # 确保当前值也被切割
            val = xValues[head:tail, :]
            allNan = np.all(nanIdx[head:tail, :], axis=0)
            maxPos = val.shape[0] - np.nanargmin(val, axis=0)
            maxPos[allNan] = -1
            raw[dumi, :] = maxPos
        raw[raw < 0] = np.nan
        raw[nanIdx] = np.nan        # 保留原始数据中的NaN
        raw = pd.DataFrame(raw, index=x.index, columns=x.columns)
        return raw

    @classmethod
    def TsToMax(cls, x, num, minobs=0):
        """
        计算 当前值 距离窗口期内最大值之间的间隔数, 当前值本身也算一位
        :param x:
        :param num:
        :param minobs:
        :return:
        """
        xValues = x.values
        nanIdx = np.isnan(xValues)
        xValues[nanIdx] = 0     # 为正常调用 nanargmax, 把空值设为0
        raw = np.full(x.shape, np.nan)
        for dumi in range(minobs, x.shape[0]):
            head = max(dumi - num + 1, 0)
            tail = dumi + 1     # 确保当前值也被切割
            val = xValues[head:tail, :]
            allNan = np.all(nanIdx[head:tail, :], axis=0)
            maxPos = val.shape[0] - np.nanargmax(val, axis=0)
            maxPos[allNan] = -1
            raw[dumi, :] = maxPos
        raw[raw < 0] = np.nan
        raw[nanIdx] = np.nan        # 保留原始数据中的NaN
        raw = pd.DataFrame(raw, index=x.index, columns=x.columns)
        return raw

    @classmethod
    def FindRank(cls, x, num, minobs=0, pct=False):
        """
        计算 当前值在过去n天的顺序排位 最小值排名为1
        注： pandas.DataFrame.rolling.allpy 是一列列的计算，因而效率较低，不如直接按行循环
        :param x:
        :param num:     窗口长度
        :param minobs:  窗口需要的最小样本数，不足则结果为 NaN
        :param pct:     返回比例值：及当前值在过去num天的分位数, 总样本数将忽略NaN值
        :return:        返回结果中， NaN 值将被忽略  ex. [1,nan ,3] 的FindRank 值为 2
        """
        xValues = x.values
        nanIdx = np.isnan(xValues)
        raw = np.full(x.shape, np.nan)
        for dumi in range(minobs, x.shape[0]):
            head = max(dumi - num + 1, 0)
            tail = dumi + 1     # 确保当前值也被切割
            val = xValues[head:tail, :]
            vrk = np.sum(val <= val[-1, :], axis=0)
            raw[dumi, :] = vrk
        if pct:
            valids = 1 - nanIdx.astype(np.int)
            valCnt = np.zeros(valids.shape)
            valCnt[:num, :] = np.cumsum(valids[:num, :], axis=0)
            for dumi in range(num, x.shape[0]):
                valCnt[dumi, :] = valCnt[dumi-1, :] + valids[dumi, :] - valids[dumi-num, :]
            valCnt[valCnt == 0] = np.nan       # 避免除以0的情况
            raw = raw / valCnt
        raw[nanIdx] = np.nan        # 保留原始数据中的NaN
        raw = pd.DataFrame(raw, index=x.index, columns=x.columns)
        return raw

    @classmethod
    def Rank(cls, x):
        # 注 不应再时间序列排序中使用该函数，应用于截面排序
        # method : {‘average’, ‘min’, ‘max’, ‘first’, ‘dense’}
        # na_option : {‘keep’, ‘top’, ‘bottom’}
        raw = x.rank(method='average', na_option='bottom', pct=False, ascending=True, axis=1)
        return raw

    @classmethod
    def Corr(cls, x, y, num, minobs=2):
        nanPos = x.isnull().values + y.isnull().values  # x 和 y 需要全部有值才有意义
        cpX = x.copy(deep=True)
        cpY = y.copy(deep=True)
        cpX[nanPos] = np.nan
        cpY[nanPos] = np.nan
        numerator = CalculatorPanel.Mean(cpX * cpY, num, minobs) - CalculatorPanel.Mean(cpX, num, minobs) * CalculatorPanel.Mean(cpY, num, minobs)
        denominator = CalculatorPanel.td(cpX, num, ddof=0, minobs=minobs) * CalculatorPanel.Std(cpY, num, ddof=0, minobs=minobs)
        return numerator / denominator

    @classmethod
    def AlphaBetaSigma(cls, x, y, num, calcAlpha=False, calcSigma=False, minobs=2):
        nanPos = x.isnull().values + y.isnull().values  # x 和 y 需要全部有值才有意义
        cpX = x.copy(deep=True)
        cpY = y.copy(deep=True)
        cpX[nanPos] = np.nan
        cpY[nanPos] = np.nan
        numerator = CalculatorPanel.Mean(cpX * cpY, num, minobs) - CalculatorPanel.Mean(cpX, num, minobs) * CalculatorPanel.Mean(cpY, num, minobs)
        denominator =CalculatorPanel .Var(cpX, num, ddof=0, minobs=minobs)
        beta = numerator / denominator
        if calcAlpha or calcSigma:  # 需要计算 alpha
            meanX = CalculatorPanel.Mean(x=cpX, num=num, minobs=minobs)
            meanY = CalculatorPanel.Mean(x=cpY, num=num, minobs=minobs)
            alpha = meanY - meanX * beta
            if calcSigma:  # 需要 计算残差平方和/ valid data num
                ssX = CalculatorPanel.Mean(x=cpX ** 2, num=num, minobs=minobs)
                ssY = CalculatorPanel.Mean(x=cpY ** 2, num=num, minobs=minobs)
                sXY = CalculatorPanel.Mean(x=cpX * cpY, num=num, minobs=minobs)
                sigma = ssY + ssX * beta ** 2 + alpha ** 2 - 2 * beta * sXY - 2 * alpha * meanY + 2 * alpha * beta * meanX
                return alpha, beta, sigma
            else:
                return alpha, beta
        else:
            return beta

    @classmethod
    def RegBeta(cls, x, y, num):
        raw =CalculatorPanel.AlphaBetaSigma(x=x, y=y, num=num)
        return raw

    @classmethod
    def RegAlpha(cls, x, y, num):
        alpha, beta = CalculatorPanel.AlphaBetaSigma(x=x, y=y, num=num, calcAlpha=True)
        return alpha

    @classmethod
    def RegResi(cls, x, y, num):
        alpha, beta, sigma = CalculatorPanel.AlphaBetaSigma(x=x, y=y, num=num, calcAlpha=True, calcSigma=True)
        return sigma
