import requests
import StringIO
import zipfile
import numpy as np
import pandas as pd # pandas
import matplotlib.pyplot as plt # module for plotting

# If this module is not already installed, you may need to install it.
# You can do this by typing 'pip install seaborn' in the command line
import seaborn as sns

import sklearn
import sklearn.datasets
import sklearn.decomposition
import sklearn.neighbors
import sklearn.metrics

def problem1():
    teams = pd.read_csv("./data/Teams.csv")
    players = pd.read_csv("./data/Batting.csv")
    salaries = pd.read_csv("./data/Salaries.csv")
    fielding = pd.read_csv("./data/Fielding.csv")
    master = pd.read_csv("./data/Master.csv")

    ### Your code here ###
    print "Dimensions of teams DataFrame:", teams.shape
    print "Dimensions of players DataFrame:", players.shape
    print "Dimensions of salaries DataFrame:", salaries.shape
    print "Dimensions of fielding DataFrame:", fielding.shape
    print "Dimensions of master DataFrame:", master.shape

    print salaries.columns

    id_groups = salaries.groupby(by=['playerID'])['playerID', 'salary'].mean()
    median_salaries = pd.merge(master[['playerID', 'nameFirst', 'nameLast']], id_groups, left_on='playerID', right_index=True, how='inner')
    print median_salaries.shape

    print teams.sample(2)
    sub_teams = teams[(teams['G']==162) & (teams['yearID']>1947)].copy()
    print sub_teams.shape
    sub_teams["1B"] = sub_teams.H - sub_teams["2B"] - sub_teams["3B"] - sub_teams["HR"]
    sub_teams["PA"] = sub_teams.BB + sub_teams.AB

    for col in ["1B", "2B", "3B", "HR", "BB"]:
        sub_teams[col] = sub_teams[col] / sub_teams.PA

    stats = sub_teams[["teamID", "yearID", "W", "1B", "2B", "3B", "HR", "BB"]].copy()
    print stats.shape

    # for col in ["1B", "2B", "3B", "HR", "BB"]:
    #     plt.scatter(stats.yearID, stats[col], c="g", alpha=0.5)
    #     plt.title(col)
    #     plt.xlabel('Year')
    #     plt.ylabel('Rate')
    #     plt.show()

    def mean_normal(df):
        sub_rates = df[["1B", "2B", "3B", "HR", "BB"]]
        df[["1B", "2B", "3B", "HR", "BB"]] = sub_rates - sub_rates.mean(axis=0)
        return df

    stats = stats.groupby('yearID').apply(mean_normal)
    print stats.head(1)

    # linear fit
    from sklearn import linear_model
    clf = linear_model.LinearRegression()

    stat_train = stats[stats.yearID < 2002]
    stat_test = stats[stats.yearID >= 2002]

    XX_train = stat_train[["1B", "2B", "3B", "HR", "BB"]].values
    XX_test = stat_test[["1B", "2B", "3B", "HR", "BB"]].values

    YY_train = stat_train.W.values
    YY_test = stat_test.W.values

    clf.fit(XX_train, YY_train)

    print 'mse=%.2f' % (np.mean((YY_test - clf.predict(XX_test)) ** 2  ))

    # plt.plot(YY_test, color='r')
    # plt.plot(clf.predict(XX_test))
    # plt.show()

    # problem_1f
    subPlayers = players[(players.AB + players.BB > 500) & (players.yearID > 1947)].copy()

    subPlayers["1B"] = subPlayers.H - subPlayers["2B"] - subPlayers["3B"] - subPlayers["HR"]
    subPlayers["PA"] = subPlayers.BB + subPlayers.AB

    for col in ["1B", "2B", "3B", "HR", "BB"]:
        subPlayers[col] = subPlayers[col] / subPlayers.PA

    # Create playerstats DataFrame
    playerstats = subPlayers[["playerID", "yearID", "1B", "2B", "3B", "HR", "BB"]].copy()
    playerstats = playerstats.groupby(by=['playerID']).apply(mean_normal)

    def meanNormalizePlayerLS(df):
        df = df[['playerID', '1B', '2B', '3B', 'HR', 'BB']].mean()
        return df

    def getyear(x):
        return int(x[0:4])

    playerLS = playerstats.groupby('playerID').apply(meanNormalizePlayerLS).reset_index()
    playerLS = pd.merge(master[["playerID","debut","finalGame"]], playerLS, how='inner', on="playerID")
    print playerLS.shape
    print playerLS.columns
    playerLS["debut"] = playerLS.debut.apply(getyear)
    playerLS["finalGame"] = playerLS.finalGame.apply(getyear)
    cols = list(playerLS.columns)
    cols[1:3] = ["minYear", "maxYear"]
    playerLS.columns = cols
    avg_rates = playerLS[['1B', '2B', '3B', 'HR', 'BB']].values
    playerLS['OPW'] = clf.predict(avg_rates)
    print playerLS.sample(1)
    print 'done!!!'

def main():
    problem1()


if __name__ == '__main__':
    main()