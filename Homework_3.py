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
    stats.head()

def main():
    problem1()


if __name__ == '__main__':
    main()