import math


# kullback entroy
def calKL(p, q):
    sum = 0.0

    for i in range(len(p)):
        print i, p[i], q[i]
        sum += p[i] * math.log(p[i]/q[i])

    return sum


if __name__ == '__main__':
    p = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    q = [0.01, 0.38, 0.23, 0.05, 0.19, 0.22, 0.001, 0.63]
    print calKL(p, q)
    print 'hello world'
