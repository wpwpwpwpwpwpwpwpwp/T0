"""
截面收益计算
输入x和y均为一列的数组形式, 且已处理过nan值
"""
import numpy as np


def IC(x, y):
    """Pearson 相关系数"""
    x_ = x - x.mean(axis=0)
    y_ = y - y.mean(axis=0)
    xy = x_ * y_
    x2 = x_ ** 2
    y2 = y_ ** 2
    return xy.sum(axis=0) / np.sqrt(x2.sum(axis=0) * y2.sum(axis=0))


def RKIC(x, y):
    """Spearman 相关系数"""
    xNew = np.argsort(np.argsort(x))
    yNew = np.argsort(np.argsort(y))
    x_ = xNew - xNew.mean(axis=0)
    y_ = yNew - yNew.mean(axis=0)
    xy = x_ * y_
    x2 = x_ ** 2
    y2 = y_ ** 2
    return xy.sum(axis=0) / np.sqrt(x2.sum(axis=0) * y2.sum(axis=0))


def BETA(x, y):
    """单因子回归斜率"""
    x_ = x - x.mean(axis=0)
    y_ = y - y.mean(axis=0)
    xy = x_ * y_
    x2 = x_ ** 2
    return xy.sum(axis=0) / x2.sum(axis=0)


def GPIC(x, y, groupNum):
    """分n组后的相关系数"""
    xRank = np.argsort(np.argsort(x))  # 计算序号
    xQuantile = xRank / len(x)  # 计算分位数
    groupIDs = np.array(range(1, groupNum + 1))  # 设置1-groupNum的组号
    groupMeanY = np.zeros([groupNum])  # 设置groupNum大小的空间存放每组平均收益
    xGroupRank = np.ceil(xQuantile * groupNum)  # 确定每个x值分别属于哪一组
    for dumi in range(1, groupNum + 1):
        groupIdx = xGroupRank == dumi  # 提取dumi组信息
        groupMeanY[dumi - 1] = np.sum(groupIdx * y, axis=0) / np.sum(groupIdx, axis=0)  # 计算该组平均收益率
    return IC(groupIDs, groupMeanY)


def TBDF(x, y, cut):
    """top平均收益 - bottom平均收益"""
    xRank = np.argsort(np.argsort(x))  # 计算序号
    xQuantile = xRank / len(x)  # 计算分位数
    topIdx = xQuantile >= 1 - cut  # 筛选出top组
    topRet = (y * topIdx).sum(axis=0) / topIdx.sum(axis=0)  # 计算top组平均收益
    botIdx = xQuantile < cut  # 筛选出bottom组
    botRet = (y * botIdx).sum(axis=0) / botIdx.sum(axis=0)  # 计算bottom组平均收益
    indiResult = topRet - botRet
    return indiResult