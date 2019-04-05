# encoding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def test_window(data):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
    cdata = data[(data['cookie_clk']<1000) & (data['cookie_age']<86400*365*10)]

    ax1.plot(cdata['cookie_clk'].fillna(0).rolling(50, center=True).mean())
    ax1.plot(cdata['cookie_clk'].fillna(0).expanding(50, center=True).mean())
    ax1.plot(cdata['cookie_clk'].fillna(0).ewm(50).mean(), color='black')
    # print s.rolling(3).median()
    # print s.rolling(3).mean(), s.rolling(3).std(), s.rolling(3).kurt(), s.rolling(3).skew(),
    # # 协方差、相关系数
    # ax2.plot(cdata['cookie_clk'].rolling(50).cov(cdata['cookie_age']))
    ax2.plot(cdata['cookie_clk'].rolling(50).corr(cdata['cookie_age']))
    ax2.plot(cdata['cookie_clk'].rolling(50).corr(cdata['cookie_age']))

    plt.show()

def test_group(data):
    grouped = data.groupby(['ipc'], sort=False)
    print grouped.size()

    cut_points = [-1, 4, 7, 11, 16, 22, 32, 51, 12510]
    (outs, bins) = pd.cut(data['cookie_clk'].dropna(), cut_points, right=False, retbins=True)
    print data.groupby(outs).size()

    # def mapping(ipc):
    #     ips = ipc.split(".")
    #     if len(ips) == 3:
    #         return "%s.%s" % (ips[0], ips[1])
    #     return ""
    #
    # grouped = data.groupby(mapping(data['ipc']), axis=0)
    # print grouped

    # 遍历
    for key, value in grouped:
        print key
        break
    value2 = grouped.get_group('111.19.96')

    def af_sum(list):
        return sum(list)
    # agg
    print grouped['cookie_clk'].agg([np.sum, np.mean, af_sum]).head(1)
    # transform
    grouped.ffill()
    data['zscore'] = grouped['cookie_clk'].transform(lambda x: (x-x.mean()/x.std()))
    from scipy.stats import zscore
    data['zscore2'] = grouped['cookie_clk'].apply(zscore)
    print grouped.get_group('180.130.2')[['cookie_clk', 'zscore', 'zscore2']]
    # print grouped['cookie_clk'].rolling(2).mean()# cumsum()
    # filtration
    # print grouped.filter(lambda x: len(x)>20)['ipc'].head(1)

    # apply
    print grouped['cookie_clk'].apply(lambda x: (x-x.min())/x.max())

def test_cut():
    data = pd.Series(np.random.randn(100))
    factor = pd.qcut(data, [0, .25, .5, .75, 1.])
    print data.groupby(factor).size()

def main():
    data = pd.read_csv("/Users/pengguo/Documents/workspace/ml/statistics/data/filter3_feas_dis_table_native"
                       , sep=chr(1)
                       # , dtype = {'a': np.int32}
                       , names=['timestamp', 'mid', 'g_referer', 'g_turl', 'g_searchcontent', 'g_qrresult', 'g_custid'
                                , 'click_step', 'empty_wangwang', 'cookie_age', 'stay_time', 'cookie_clk', 'acookie_search'
                                , 'acookie_pv', 'acookie_bug', 'acookie_item', 'acookie_home', 'acookie_r_ctr'
                                , 'ipc', 'search_keyword', 'adgroup_id', 'nick', 'pos', 'cate1', 'cate3'
                                , 'is_filter', 'bf_code', 'is_deal']
                       )
    # test_window(data.head(10000))
    test_group(data.head(50))
    # test_cut()

if __name__ == '__main__':
    main()