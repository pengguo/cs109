# encoding=utf-8

import numpy as np
import matplotlib.pyplot as plt

def main():
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    X = data.data
    Y = data.target
    print Y[:4]

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    com_data = pca.fit_transform(X)

    L1 = [e[0] for e in com_data]
    L2 = [e[1] for e in com_data]


    from sklearn.cluster import KMeans
    # training
    clf = KMeans(n_clusters=2)
    clf.fit(X)
    clu_2 = clf.predict(X)

    clf = KMeans(n_clusters=3)
    clf.fit(X)
    clu_3 = clf.predict(X)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.scatter(L1, L2, c=clu_2, marker='s')
    ax2.scatter(L1, L2, c=clu_3, marker='s')
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()