# encoding=utf-8

# https://nbviewer.jupyter.org/github/cs109/2014/blob/master/homework/HW1.ipynb

import requests
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def download_data(url):
    response = requests.get(url)
    print response.text
    print response.content
    print response.url
    print response.encoding
    print response.status_code

def load_data_from_csv(zip_path):
    zfiles = zipfile.ZipFile(zip_path)
    zfiles.extractall('./data/')
    salary_frame = pd.read_csv('./data/Salaries.csv', sep=',')
    print salary_frame.sample(1)
    print salary_frame.shape
    team_frame = pd.read_csv('./data/Teams.csv', sep=',')
    print team_frame.sample(1)
    print team_frame.shape

    # print salary_frame.groupby(by=['yearID', 'teamID'], as_index=False).sum()

    union_frame = pd.merge(salary_frame, team_frame, how='inner', on=['teamID', 'yearID'])
    print union_frame.shape
    return union_frame

def draw_w_salary(union_frame):
    # print union_frame.columns

    # print groups['salary'].agg(np.sum)
    groups = union_frame.groupby(by=['yearID', 'teamID'])
    for year in ['2010', '2011', '2012', '2013']:
        for (yearID, teamID), value in groups:
            if str(yearID) == year:
                cr = 'r' if str(teamID) == 'OAK' else 'b'
                plt.scatter(value['salary'].sum(), value['W'].sum(), color=cr)
        plt.title(str(year))
        plt.show()
        plt.close()

# Y = X*m + c
def solve_least_squar(union_frame):
    groups = union_frame.groupby(by=['yearID', 'teamID'])
    Y = groups['W'].sum()
    X = groups['salary'].sum()
    A = np.array([X, np.ones(len(X))])
    print A.T.shape
    m, c = np.linalg.lstsq(A.T, Y, rcond=None)[0]
    print m, c
    plt.plot(X, Y, 'o', label='Original data', markersize=2)
    plt.plot(X, m*X + c, 'r', label='Fitted line')
    plt.legend()
    plt.show()


# 原理：https://www.zhihu.com/question/37031188/answer/411760828
def problem1():
    # load_data("http://seanlahman.com/files/database/lahman-csv_2014-02-14.zip")
    data = load_data_from_csv('/Users/pengguo/Downloads/lahman-csv_2014-02-14.zip')
    # draw_w_salary(data)
    solve_least_squar(data)

def load_data_from_excel(path):
    data = pd.read_excel(path)
    print data.head(1)

def problem2():
    countries = pd.read_csv('./data/countries.csv', sep=',')
    print countries.head(1)
    income = pd.read_excel('./data/indicator_gapminder_gdp_per_capita_ppp.xlsx', sheet_name='Data')
    print income.shape
    print income.head(1)

    income.index = income[income.columns[0]]
    print income.head(2)
    income = income.drop(income.columns[0], axis=1)
    print income.head(2)

    income.columns = map(lambda x: int(x), income.columns)
    income_t = income.transpose()
    print income_t.shape
    print income_t.head(1)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 10))

    ax1.hist(income_t.loc['2000':'2000', :].dropna(axis=1), bins=20)
    # 不能反应真实分布，非线性
    ax2.hist(np.log10(income_t.loc['2000':'2000', :].dropna(axis=1)), bins=20)
    # plt.show()

    def merge_by_year(iyear):
        income = income_t.loc[iyear:iyear, :].transpose()
        # print income_t.ix[iyear].values
        # income = pd.DataFrame(income_t.ix[iyear].values, columns=['income'])
        print income.head(2)
        print income.shape
        income['Country'] = income_t.columns

        income_info = pd.merge(income, countries, on='Country')
        income_info.columns=['Income', 'Country', 'Region']
        print income_info.sample(1)
        return income_info

    y2000 = merge_by_year(2000)
    y2000.boxplot(column=['Income'], by=['Region'], rot=90, figsize=(6, 10))
    plt.ylim(10**2, 10.5**5)
    plt.show()

from scipy import stats

def get_merge_data(iyear):
    countries = pd.read_csv('./data/countries.csv', sep=',')
    income = pd.read_excel('./data/indicator_gapminder_gdp_per_capita_ppp.xlsx', sheet_name='Data')
    income.index = income[income.columns[0]]
    income = income.drop(income.columns[0], axis=1)

    income.columns = map(lambda x: int(x), income.columns)
    income_t = income.transpose()
    income = income_t.loc[iyear:iyear, :].transpose()
    income['Country'] = income_t.columns

    income_info = pd.merge(income, countries, on='Country')
    income_info.columns = ['Income', 'Country', 'Region']
    return income_info

def ration_normals(diff=1, a=2):
    X = stats.norm(loc=diff, scale=1)
    Y = stats.norm(loc=0, scale=1)
    x_sample = X.rvs(size=10000) # 随机生产符合正态分布的
    print x_sample.mean(), x_sample.std()
    print stats.norm.fit(x_sample)
    # plt.hist(x_sample, bins=10, density=True)
    x_label = np.linspace(X.ppf(0.01), X.ppf(0.99), 100) # 概率密度函数pdf的反函数ppf
    # plt.plot(x_label, X.pdf(x_label))
    # plt.plot(x_label, X.cdf(x_label)) # 累计概率密度函数 P(x<a)
    # plt.plot(x_label, X.sf(x_label), label='x_normal') # 累计概率密度函数 P(x>a)
    # plt.plot(x_label, Y.sf(x_label), label='y_normal')
    # plt.legend()
    # plt.show()
    return X.sf(a)/Y.sf(a)

def problem3_a():
    x_label = np.linspace(0, 5, 50)
    a = [2, 2.1, 3]
    for ia in a:
        plt.plot(x_label, [ration_normals(diff=x, a=ia) for x in x_label], label='a=%s' % (ia))
    plt.legend
    plt.show()

def problem3_b():
    y2012 = get_merge_data(2012)
    y2012_p =  y2012[y2012['Region'].str.upper().isin(['ASIA', 'SOUTH AMERICA'])]
    groups = y2012_p.groupby(by=['Region'])
    for key, value in groups:
        print key, value['Income'].mean()
        floc, fscale = stats.norm.fit(value['Income'])
        X = stats.norm(loc=floc, scale=fscale)
        print X.sf(10000)


def problem3():
    # problem3_a()
    problem3_b()

def main():
    # problem1()
    # problem2()
    problem3()

if __name__ == '__main__':
    main()