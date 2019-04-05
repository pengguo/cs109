# encoding=utf-8
import numpy as np
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt

def get_help():
    print stats.norm.__doc__


def test_normal():
    print stats.norm.pdf([-1, 0, 1])
    print stats.norm.cdf([-1, 0, 1])
    print stats.norm.ppf(0.5) # reverse cdf
    mean, var, skew, kurt = stats.norm.stats(moments='mvsk')
    print mean, var, skew, kurt
    fig, ax = plt.subplots(1, 1)
    X = np.linspace(stats.norm.ppf(0.01), stats.norm.ppf(0.99), 100)
    ax.plot(X, stats.norm.pdf(X), 'r-', lw=5, alpha=0.6, label='norm pdf')
    rv = stats.norm()
    ax.plot(X, rv.pdf(X), 'k-', lw=2, label='frozen pdf')
    vals = stats.norm.ppf(0.001, 0.5, 0.999)
    print np.allclose([0.001, 0.5, 0.999], stats.norm.cdf(vals))

    r = stats.norm.rvs(size=1000)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    plt.show()
'''
名称	含义
beta	beta分布
f	F分布
gamma	gam分布
poisson	泊松分布
hypergeom	超几何分布
lognorm	对数正态分布
binom	二项分布
uniform	均匀分布
chi2	卡方分布
cauchy	柯西分布
laplace	拉普拉斯分布
rayleigh	瑞利分布
t	学生T分布
norm	正态分布
expon	指数分布
'''
def create_norm():
    # 生成服从指定分布的随机数
    print stats.norm.rvs(loc=5, scale=2, size=10)
    # print stats.norm.rvs(loc=5, scale=2, size=(2, 3))
    # 求概率密度函数指定点的函数值
    print stats.norm.pdf(0, loc=0, scale=1)

    print stats.beta()

def test_dis():
    print stats.t.isf([0.1, 0.5, 0.01], [[10], [11]])

def test_fit():
    x = np.arange(1, 16, 1)
    num = [4.00, 5.20, 5.900, 6.80, 7.34,
           8.57, 9.86, 10.12, 12.56, 14.32,
           15.42, 16.50, 18.92, 19.58, 20.00]
    y = np.array(num)
    f1 = np.polyfit(x, y, 3)
    p1 = np.poly1d(f1)
    print p1

    yvals = p1(x) # estimate y
    plt.plot(x, y, 's', label='original value')
    plt.plot(x, yvals, 'r', label='polyfit value')


    def func(x, a, b):
        return a * np.exp(b/x)
    popt, pcov = optimize.curve_fit(func, x, y)
    a = popt[0]
    b = popt[1]
    yvals = func(x, a, b)
    plt.plot(x, yvals, 'g', label='curve_fit value')

    plt.show()

def test_fit2():
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    # define the data to be fit with some noise
    xdata = np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3, 0.5)
    y_noise = 0.2 * np.random.normal(scale=0.5, size=xdata.size)
    ydata = y + y_noise
    plt.plot(xdata, ydata, 'b-', label='data')

    # Fit for the parameters a, b, c of the function `func`
    popt, pcov = optimize.curve_fit(func, xdata, ydata)
    plt.plot(xdata, func(xdata, *popt), 'r-', label='fit')

    popt, pcov = optimize.curve_fit(func, xdata, ydata, bounds=(0, [3., 2., .3]))
    plt.plot(xdata, func(xdata, *popt), 'g--', label='fit-with-bounds')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def test_zscore():
    from scipy.stats import zscore
    import pandas as pd

    x = np.arange(0, 5, 1)
    y = np.arange(1, 6, 1)
    df = pd.DataFrame([x, y])
    print(df)

    df = df.apply(zscore, axis=1)
    print(df)

def main():
    get_help()
    test_zscore()
    # test_fit2()
    # create_norm()
    # test_normal()
    # test_dis()

if __name__ == '__main__':
    main()