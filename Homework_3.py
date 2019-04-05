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
    meadin_salaries = pd.merge

def main():
    problem1()


if __name__ == '__main__':
    main()