import datetime
import numpy as np

from collections import OrderedDict


def parser_cnt(filepath):
    pid_cnt = {}
    pre_pid = None
    for line in open(filepath):
        pairs = line.strip().split(",")
        if len(pairs) != 3:
            continue
        (pid, ds, cnt) = pairs
        if pre_pid is None:
            pre_pid = pid

        if pre_pid != pid:
            # print pre_pid
            # print ",".join(day_cnt)
            pid_cnt[pre_pid] = day_cnt
            global day_cnt
            day_cnt = [0] * 13
            pre_pid = pid

        idx = (datetime.datetime.strptime(ds, "%Y%m%d")-datetime.datetime.strptime("20170610", "%Y%m%d")).days
        # print  idx, type(idx)
        day_cnt[idx] = int(cnt)

    return pid_cnt

def cov(X, Y):
    print X.mean(), X.std()
    sum = 0.0
    for idx in range(len(X)):
        sum += (X[idx] - X.mean()) * (Y[idx] - Y.mean())
    sum = sum / len(X)
    print sum, sum/(X.std()*Y.std())


def test1():
    # import mars.tensor as mt
    import numpy as mt

    a = mt.arange(10).reshape(5, 2)
    a2 = mt.zeros(shape=(5, 2))
    print a.shape
    print a
    b = a[:, mt.newaxis, :] - a
    print a[:, mt.newaxis, :]
    print b.shape
    print b
    print b**2
    r = mt.triu(mt.sqrt(b ** 2).sum(axis=2))

def test2():
    xx = np.arange(4).reshape(4, 1)
    y = np.ones(5)

    print xx
    print y

    print xx+y


def main():
    test1()



if __name__ == "__main__":
    main()
    #
    # filepath = "/Users/pengguo/Documents/workspace/ml/pid_ds_cnt_sorted"
    # pid_cnt = parser_cnt(filepath)
    #
    # X = np.array(pid_cnt['430042_1006'])
    # columns = ['430042_1006']
    # for k in pid_cnt:
    #     columns.append(k)
    #     X = np.row_stack((X, pid_cnt[k]))
    #
    # print X.shape
    # print columns
    # data1 = np.corrcoef(X)[2]
    # for i in range(len(columns)):
    #     if data1[i] > 0.8:
    #         print columns[i],data1[i]


