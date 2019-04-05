
import numpy as np

def gini_coef(wealths):
    cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    print cum_wealths
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths)-1)
    print xarray
    yarray = cum_wealths / sum_wealths
    B = np.trapz(yarray, x=xarray)
    print B
    A = 0.5 - B
    return A / (A+B)


if __name__ == "__main__":
    # wealths = [100,200,323,13313,554534,645,4,3424,7,78,567,9867,67,45,97,3232331313311]
    wealths = [1,2,3]
    print gini_coef(wealths)