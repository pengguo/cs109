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


# https://www.cnblogs.com/pinard/p/6061661.html


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

    # problem 1j
    from collections import defaultdict
    def find_pos(df):
        positions = df.POS
        d = defaultdict(int)
        for pos in positions:
            d[pos] += 1
        result = max(d.iteritems(), key=lambda x: x[1])
        return result[0]

    positions_df = fielding.groupby("playerID").apply(find_pos)
    positions_df = positions_df.reset_index()
    positions_df = positions_df.rename(columns={0: "POS"})

    playerLS_merged = pd.merge(positions_df, playerLS, how='inner', on="playerID")
    playerLS_merged = pd.merge(playerLS_merged, median_salaries, how='inner', on=['playerID'])
    active = playerLS_merged[(playerLS_merged["minYear"] <= 2002) & \
                             (playerLS_merged["maxYear"] >= 2003) & \
                             (playerLS_merged["maxYear"] - playerLS_merged["minYear"] >= 3)]
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(active.salary / 10 ** 6, active.OPW, alpha=0.5, c='red')
    ax.set_xscale('log')
    ax.set_xlabel('Salary (in Millions) on log')
    ax.set_ylabel('OPW')
    ax.set_title('Relationship between Salary and Predicted Number of Wins')
    plt.show()

    print 'problem1 done!!!'

def computeTestScores(test_x, test_y, clf, cv):
    kFolds = sklearn.model_selection.KFold(n_splits=3, shuffle=False)

    scores = []
    for _, test_index in kFolds.split(test_x):
        test_data = test_x[test_index]
        test_labels = test_y[test_index]
        scores.append(sklearn.metrics.accuracy_score(test_labels, clf.predict(test_data)))
    return scores

def problem2():
    # load the iris data set
    iris = sklearn.datasets.load_iris()
    X = iris.data
    Y = iris.target
    print X.shape, Y.shape

    # put test data aside
    X_train, X_test, Y_train, Y_test = \
        sklearn.model_selection.train_test_split(X, Y, test_size=0.33, random_state=42)
    print X_train.shape, Y_train.shape

    # make a scatter plot of the data in two dimensions
    svd = sklearn.decomposition.TruncatedSVD(n_components=2)
    X_train_centered = X_train - np.mean(X_train, axis=0)
    X_2d = svd.fit_transform(X_train_centered)

    # sns.set_style('white')
    # plt.scatter(X_2d[:, 0], X_2d[:, 1], c=Y_train, s=50, cmap=plt.cm.prism)
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.title('First two PCs using iris data')
    # plt.show()
    # use cross validation to find the optimal value for k
    k = np.arange(20) + 1

    parameters = {'n_neighbors': k}
    knn = sklearn.neighbors.KNeighborsClassifier()
    clf = sklearn.model_selection.GridSearchCV(knn, parameters, cv=10)
    clf.fit(X_train, Y_train)

    print clf.best_score_, clf.best_params_

    scores = computeTestScores(test_x=X_test, test_y=Y_test, clf=clf, cv=5)
    print np.mean(scores), np.std(scores)

    print 'problem2 done!!!'

def problem3():
    digits = sklearn.datasets.load_digits()

    X = digits.data
    Y = digits.target

    print X.shape, Y.shape
    ### Your code here ###

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.33, random_state=42)

    X_train_means = np.mean(X_train, axis=0)
    X_train_centered = X_train - X_train_means
    X_test_centered = X_test - X_train_means

    print X_train.shape
    print X_test.shape
    print Y_train.shape
    print Y_test.shape

    svd = sklearn.decomposition.TruncatedSVD(n_components=2)
    X_2d = svd.fit_transform(X_train_centered)

    # fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(5,8))
    # ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=Y_train, s=50, cmap=plt.cm.Paired)
    # # plt.colorbar()
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.title('First two PCs using digits data')
    # plt.show()

    # ind = np.logical_or(Y_train == 4, Y_train == 1)
    # ax2.scatter(X_2d[ind, 0], X_2d[ind, 1], c=Y_train[ind], s=50, cmap=plt.cm.Paired)
    # # plt.show()

    # fit knn
    k = np.arange(20) + 1
    parameters = {'n_neighbors': k}
    knn = sklearn.neighbors.KNeighborsClassifier()
    clf = sklearn.model_selection.GridSearchCV(knn, parameters, cv=10)
    all_scores = []
    all_k = []
    all_d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for d in all_d:
        print d
        svd = sklearn.decomposition.TruncatedSVD(n_components=d)
        X_d = svd.fit_transform(X_train_centered)
        X_d_test = svd.transform(X_test_centered)

        clf.fit(X_d, Y_train)

        all_scores.append(computeTestScores(test_x=X_d_test, test_y=Y_test, clf=clf, cv=10))
        all_k.append(clf.best_params_['n_neighbors'])
    print all_scores
    print all_k

def main():
    # problem1()
    # problem2()
    problem3()


if __name__ == '__main__':
    main()