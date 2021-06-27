import numpy as np
import math
import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import torch
import dgl


def coord_polar_to_cart(a, b):
    '''
    极坐标转直角坐标
    :param a:
    :param b:
    :return:
    '''
    x = a * np.cos(b)
    y = a * np.sin(b)
    return x, y


def coord_cart_to_polar(x, y):
    '''
    直角坐标转极坐标
    :param x:
    :param y:
    :return:
    '''
    a = np.sqrt(x**2 + y**2)
    b = np.arctan(y / x)
    if x < 0:
        b += np.pi
    elif x > 0 and y < 0:
        b += np.pi * 2
    return a, b


class _DataItemGen:
    '''
    外面一个大圆，里面一个小圆，有一堆点
    数据：每个点的XY坐标
    标签：若是大圆则为1，若是小圆则为0
    '''

    def __init__(self, r1=10, r2=5, deg_add=2, offset=(0, 0)):
        pts = []
        cls = []
        for d in range(0, 360, deg_add):
            pt1 = coord_polar_to_cart(r1, np.deg2rad(d))
            pt2 = coord_polar_to_cart(r2, np.deg2rad(d))
            pts.append(pt1)
            cls.append(1)
            pts.append(pt2)
            cls.append(0)
        pts = np.float32(pts) + offset
        cls = np.int32(cls)

        self.pts = pts
        self.cls = cls

        # 构造稠密邻接矩阵，每个节点均与周围节点链接，
        nei = 1 - np.eye(pts.shape[0], dtype=np.uint8)
        self.nei = nei

    def get_graph(self):
        g = nx.DiGraph()
        for i, (pt, c) in enumerate(zip(self.pts, self.cls)):
            g.add_node(i, pos=pt, cls=c)
        for u, li in enumerate(self.nei):
            for v, b in enumerate(li):
                if b != 0:
                    g.add_edge(u, v)
        return g


class DatasetReader:
    def __init__(self):
        cache_file = 'data.pkl'
        if not os.path.isfile(cache_file):
            train_data = self.gen_data(500)
            valid_data = self.gen_data(300)
            test_data = self.gen_data(300)
            d = {
                'train_data': train_data,
                'valid_data': valid_data,
                'test_data': test_data,
            }
            pickle.dump(d, open(cache_file, 'wb'))

        d = pickle.load(open(cache_file, 'rb'))
        self.train_data = d['train_data']
        self.valid_data = d['valid_data']
        self.test_data = d['test_data']

    @staticmethod
    def gen_data(n):
        ds = []
        for _ in range(n):
            center = np.random.randint([-100, -100], [100, 100])
            deg_add = np.random.randint(2, 45)
            r1 = np.random.randint(5, 50)
            r2 = np.random.randint(1, r1-1)
            dg = _DataItemGen(r1, r2, deg_add, center)
            d = dg.get_graph()
            g = dgl.from_networkx(d, ['pos', 'cls'])
            g.ndata['pos'] = g.ndata['pos'].type(torch.float32)
            g.ndata['cls'] = g.ndata['cls'].type(torch.long)
            ds.append(g)
        return ds

    def train_batch_gen(self, batch_size=10, batch_count=100):
        for _ in range(batch_count):
            ids = np.random.randint(0, len(self.train_data), batch_size)
            ds = [self.train_data[i] for i in ids]
            # r = dgl.batch(ds)
            yield ds

    def valid_batch_gen(self, batch_size=10):
        cur_data = self.valid_data
        n = int(math.ceil(len(cur_data) / batch_size))
        for i in range(n):
            ss = [batch_size*i, batch_size*(i+1)]
            # ds = [g.clone() for g in cur_data[ss[0]: ss[1]]]
            ds = cur_data[ss[0]: ss[1]]
            # r = dgl.batch(ds)
            yield ds

    def test_batch_gen(self, batch_size=10):
        cur_data = self.test_data
        n = int(math.ceil(len(cur_data) / batch_size))
        for i in range(n):
            ss = [batch_size*i, batch_size*(i+1)]
            # ds = [g.clone() for g in cur_data[ss[0]: ss[1]]]
            ds = cur_data[ss[0]: ss[1]]
            # r = dgl.batch(ds)
            yield ds


if __name__ == '__main__':
    # 生成的图的可视图
    dg = _DataItemGen(deg_add=45)
    g = dg.get_graph()
    # g = nx.from_numpy_array(dg.nei)
    ax = plt.figure(111, (6., 6.), dpi=100)
    # nx.draw(g, pos=dg.pts, node_size=50, node_color=[[.5, .5, .5, ]])
    nx.draw(g, pos=dg.pts, labels=dict(zip(np.arange(len(dg.cls)), dg.cls)), node_size=50)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    d = DatasetReader()

    for i, t in enumerate(d.train_batch_gen(10, 100)):
        print(i)

    for i, t in enumerate(d.valid_batch_gen(10)):
        print(i)

    for i, t in enumerate(d.test_batch_gen(10)):
        print(i)
