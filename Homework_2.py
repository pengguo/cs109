# encoding=utf-8

import datetime
import requests
from StringIO import StringIO
import numpy as np
import pandas as pd # pandas
import matplotlib.pyplot as plt # module for plotting
import datetime as dt # module for manipulating dates and times
import numpy.linalg as lin # module for performing linear algebra operations

def problem1():
    # problem 1a
    exprs = pd.read_csv('./data/exprs_GSE5859.csv', sep=',')
    exprs.index = exprs[exprs.columns[0]]
    exprs = exprs.drop(exprs.columns[0], axis=1)

    sampleinfo = pd.read_csv('./data/sampleinfo_GSE5859.csv', sep=',')

    exprs = exprs[sampleinfo.filename] # reorder exprs columns
    print (exprs.columns == sampleinfo.filename).all()

    sampleinfo.date = pd.to_datetime(sampleinfo.date)
    sampleinfo['year'] = map(lambda x : x.year, sampleinfo.date)
    sampleinfo['month'] = map(lambda x: x.month, sampleinfo.date)

    oct31 = dt.datetime(2002, 10, 31, 0, 0)
    sampleinfo['elapsedInDays'] = map(lambda x: (x - oct31).days, sampleinfo.date)
    print sampleinfo.head(3)

    # problem 1d
    sampleinfoCEU = sampleinfo[sampleinfo.ethnicity == 'CEU']
    print sampleinfoCEU.shape
    exprsCEU = exprs[sampleinfoCEU.filename]
    print (exprsCEU.columns == sampleinfoCEU.filename).all()
    print exprsCEU.head(2)
    print exprsCEU.mean(axis=1).head(2) # each sample, avg of row
    exprsCEU_normal = exprsCEU.apply(lambda x : x - exprsCEU.mean(axis=1), axis=0)
    print exprsCEU_normal.head(3)

    U,Sigma,V = np.linalg.svd(exprsCEU_normal, full_matrices=True)
    print U.shape, Sigma.shape, V.shape
    VT = V.T

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.hist(VT[:, 0], bins=25)
    ax2.scatter(sampleinfoCEU.elapsedInDays, VT[:, 0])
    plt.show()

def problem2():
    election = pd.read_csv('./data/2012-general-election-romney-vs-obama.csv', sep=',')
    election['Start Date'] = pd.to_datetime(election['Start Date'])
    print election.sample(1)
    print election.columns
    sub_ele = election[map(lambda x: (x.year==2012) and (x.month==11), election['Start Date'])]
    sub_ele.drop_duplicates('Pollster', inplace=True)
    M = len(sub_ele)
    print M
    N = sub_ele['Number of Observations'].mean()
    print N
    X = np.random.binomial(N, 0.53, size=1) # for i in range(1000)]
    X = np.random.binomial(N, 0.53, size=1000)/N  # for i in range(1000)]
    print X.mean(), X.std()
    obs = map(lambda x: np.mean(np.random.binomial(1, 0.53, size=int(N))), range(1000))

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.hist(obs, bins=25)

    from scipy import stats
    stats.probplot((obs-np.mean(obs))/np.std(obs, ddof=1), dist='norm', plot=ax2)

    plt.show()

def main():
    # problem1()
    problem2()

if __name__ == '__main__':
    main()