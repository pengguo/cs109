import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# http://python.jobbole.com/81321/
def binomialDis():
    n = 10
    p = 0.3
    k = np.arange(0, 11)
    binomial = stats.binom.pmf(k, n, p)
    plt.plot(k, binomial, 'o-')
    plt.xlabel('num of meet condition')
    plt.ylabel('Probability')
    plt.show()

def binomialSim():
    binom_sim = data = stats.binom.rvs(n=10, p=0.3, size=10000)
    print "Mean:", np.mean(binom_sim)
    print "Sd:", np.std(binom_sim, ddof=1)
    plt.hist(binom_sim, bins=10, normed=True)
    plt.xlabel("count of meet condition")
    plt.ylabel("density")
    plt.show()

def PoissonDis():
    plambda = 2
    n = np.arange(0, 10)
    y = stats.poisson.pmf(n, plambda)
    plt.plot(n, y, 'o-')
    plt.xlabel('num of meet condition')
    plt.ylabel('Probability')
    plt.show()




if __name__ == "__main__":
    # binomialDis()
    # binomialSim()
    PoissonDis()
