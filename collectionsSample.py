#coding=utf-8

from collections import namedtuple

def testNamedTuple():
    websites = [
        ('sohu', 'www.google.com', u'佩奇'),
        ('163', 'www.163.com', u'丁磊')
    ]

    websiten = namedtuple('website', ['name', 'url', 'founder'])
    for website in websites:
        website = websiten._make(website)
        print website

# https://docs.python.org/2/library/collections.html#collections.deque
from collections import deque
def testDeque():
    aque = deque()
    aque.append(2)
    aque.append(3)
    aque.appendleft(1)
    print aque
    print aque.count(3)
    aque.extend('456')
    aque.extendleft("0")
    print aque
    aque.rotate(1)
    print aque

from collections import Counter
def testCounter():
    s = '''A Counter is a dict subclass for counting hashable objects. It is an unordered collection where elements are stored as dictionary keys and their counts are stored as dictionary values. Counts are allowed to be any integer value including zero or negative counts. The Counter class is similar to bags or multisets in other languages.'''.lower()
    c = Counter(s)
    # 获取出现频率最高的5个字符
    print c.most_common(5)

from collections import OrderedDict
def testOrderDict():
    items = (
        ("A", 1),
        ("B", 2),
        ("C", 3),
    )
    order_dict = OrderedDict(items)
    for k in order_dict:
        print k,order_dict[k]


if __name__ == "__main__":
    # testNamedTuple()
    # testDeque()
    # testCounter()
    testOrderDict()