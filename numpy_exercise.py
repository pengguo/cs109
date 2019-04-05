
import numpy as np

print np.__version__
# Z = np.zeros(10)
# Z[5] = 1
# print Z
# reverse_z = Z[::-1]
# print reverse_z
#
# print np.arange(9).reshape(3,3)
# nz = np.nonzero([1,2,0,0,4,0]) # indices of non-zero
# print nz
#
# print np.eye(3)  # identity matrix
#
# # print np.random.random((3,3,3))
#
# z = np.random.random((10,10))
# print z
# print z.max(), z.min(), z.mean()
#
# z = np.ones((10,10)) # 1 on the border and 0 inside
# z[1:-1,1:-1] = 0
# print z
#
# print 0 * np.nan




def tutorial01():
    a = np.arange(15)
    b = a.reshape(3, 5)
    print b.shape
    print b.ndim
    print b.dtype.name
    print b.itemsize
    print b.size

def test_create():
    a = np.array([1,2,3,4,5])
    print a
    print np.array([(1,2,3), (4,5,6)], dtype=float)
    print np.zeros((2, 3))
    print np.ones((2, 3))
    print np.empty((2, 3))
    print np.arange(0, 10, 2)
    from numpy import pi
    x = np.linspace(0, 5 * pi, 10)
    print x
    print np.sin(x)
    print np.arange(24).reshape(2,3,4) # 3d array

    # np.set_printoptions(threshold=np.nan), force print all

def test_opt():
    b = np.arange(4)
    print b
    print b**2
    print 10*np.sin(b)
    print b<2

    a1 = np.array([[1,2,3], [4,5,6]])
    a2 = np.array([[0.1,0.2,0.3], [0.4,0.5,0.6]])
    print a1
    print a2
    print a1 + a2
    print a1 * a2
    c = np.arange(6).reshape(3,2)
    print c
    print a1.dot(c)
    a2 += a1
    print a2

def test_agg():
    d = np.random.random((2, 3))
    print 'sum:', d.sum()
    print d.min()
    a = np.arange(12).reshape(3, 4)
    print a
    print a.sum(axis = 0)
    print a.sum(axis = 1)
    print a.cumsum(axis = 1) # cumulative sum along each row

    c = np.arange(3)
    print np.exp(c)
    print np.sqrt(c)
    print np.add(c, np.arange(3))

    e = 10 * np.random.random((4, 3))
    print e
    print np.sort(e)

    print np.dot(a, e)

def test_idx():
    a = np.arange(10) ** 3
    print a
    print a[2]
    print a[0:6:2] # start:end:step
    print a[::-1] # reversed

    def f(x, y):
        return 10*x + y
    ma = np.fromfunction(f, (3, 4), dtype=np.int16)
    print ma
    print ma[1,2]
    print ma[1, 0:3]
    print ma[1, :]
    for m in ma:
        print m
    for e in ma.flat:
        pass

    idx = np.array([1, 1, 3, 5])
    print a[idx]
    idx = np.array([
        [1, 1],
        [3, 5]
    ])
    print a[idx]
    bool_idx = a>10
    print bool_idx
    print a[bool_idx]

def test_shape():
    a = np.floor(10 * np.random.random((3, 4)))
    print a
    print a.ravel() # flattened
    print a.T
    print a.reshape(4, 3)
    a.resize(4, 3)
    print a

def test_stack():
    a = np.floor(10 * np.random.random((2, 2)))
    b = np.floor(10 * np.random.random((2, 2)))

    print a
    print b
    print np.vstack((a, b))
    print np.hstack((a, b))

    c = np.array([1, 2])
    d = np.array([3, 4])
    print "c:", c
    print d
    print np.column_stack((c, d)) # # returns a 2D array
    from numpy import newaxis
    print "c: newaxis", c[: , newaxis]
    print np.column_stack((c[:, newaxis], d[:, newaxis]))

    m = np.floor(10*np.random.random((2,12)))
    print m
    print np.hsplit(m, 3)
    print np.hsplit(m, (3, 4))

def test_copy_view():
    a = np.arange(4)
    b = a
    print b is a
    b.resize((2, 2))
    print a.shape
    print id(a), id(b)

    c = a.view()
    print c is a
    print c.base is a
    print c.flags.owndata
    c[1, 1]=100
    print a
    s = a[:, 1]
    s[:] = 11 # change in a
    print a

    a_copy = a.copy()
    print a_copy is a
    a_copy[1, 1] = 100
    print a

import matplotlib.pyplot as plt
def test_hist():
    a = np.random.normal(2, 0.5, 1000)
    plt.hist(a, bins=50, density=1)
    plt.show()
    plt.close()

def build_matrix(feature_size, bin_size):
    s = np.arange(feature_size * bin_size)
    m = np.reshape(s, (feature_size, bin_size))

    for j in xrange(0, 3):
        curr_size = bin_size / 2**j
        m2 = np.sum(np.reshape(m[0], (curr_size, 2**j)), axis=1)
        print m2

def main():
    # tutorial01()
    # test_create()
    # test_opt()
    # test_agg()
    test_idx()
    # test_shape()
    # test_stack()
    # test_copy_view()
    # test_hist()
    # build_matrix(5, 16)

if __name__ == "__main__":
    main()

