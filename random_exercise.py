
import random


def test_random():
    print random.random()


def test_shuffle(src, K):
    for i in range(K):
        idx = random.randint(0, len(src)-1)
        print src[idx], len(src)
        src.pop(idx)
    pass

def test_print(alist, n, k, cur):
    if cur == k-1:
        for i in range(n):
            for j in range(cur):
                print alist[i]

    for m in range(n):
        test_print(alist, m-1, 3, cur+1)



def main():
    # src = [1,2,3,4,5,6,7,8,9,10]
    # test_shuffle(src, 5)
    # test_random()
    list = [1,2,3,4,5]
    test_print(list, 5, 3, 0)

if __name__ == '__main__':
    main()