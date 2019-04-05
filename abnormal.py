from scipy import stats

def testKL():
    proba = [0.1, 0.3, 0.5, 0.1]
    probb = [0.2, 0.3, 0.4, 0.1]
    probc = [0.5, 0.1, 0.1, 0.3]
    print stats.entropy(proba, probb)
    print stats.entropy(proba, probc)
    print myKL(proba, probb)

from math import log
def myKL(pk, qk):
    result = 0.0
    for i in range(0, len(pk)):
        result += pk[i] * (log(pk[i]/qk[i]))
    return result


if __name__ == "__main__":
    testKL()
    print 'hello world'