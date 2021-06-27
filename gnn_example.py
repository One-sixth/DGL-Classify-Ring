'''
使用DGL分类全连接图，内外环
'''


import copy
import os

import imageio
import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import SAGEConv
from g_dataset import DatasetReader


# 定义图神经网络
class GNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        n_layers = 2
        n_hidden = 64
        act = nn.ReLU()

        self.layer1 = SAGEConv(in_ch, n_hidden, 'mean', activation=None)
        self.layer1_norm = nn.BatchNorm1d(n_hidden)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            cur_layer = nn.ModuleList([
                SAGEConv(n_hidden, n_hidden, 'mean', activation=None),
                nn.BatchNorm1d(n_hidden),
                act
            ])
            self.layers.append(cur_layer)

        self.out_layer1 = SAGEConv(n_hidden, out_ch, 'mean')

    def forward(self, g, features):
        h = features
        h = self.layer1(g, h)
        h = self.layer1_norm(h)
        for i, layer in enumerate(self.layers):
            h = layer[0](g, h)
            h = layer[1](h)
            h = layer[2](h)
        h = self.out_layer1(g, h)
        return h


def norm_each_g_pos_feats(batch_g):
    # 将输入数据标准化，可以收敛更快
    for g_i, g in enumerate(batch_g):
        # 对图进行浅复制，避免对原图造成影响，又不会产生过大的复制负担
        g = copy.copy(g)
        # 对每个图的数据进行标准化
        pos_feats = g.ndata['pos']
        pos_feats = (pos_feats - pos_feats.mean(dim=0, keepdim=True)) / pos_feats.std(dim=0, keepdim=True)
        g.ndata['pos'] = pos_feats
        batch_g[g_i] = g
    return batch_g


def tr_figure_to_array(fig):
    '''
    转换 matplotlib 的 figure 到 numpy 数组
    '''
    fig.canvas.draw()
    mv = fig.canvas.buffer_rgba()
    im = np.asarray(mv)
    # 原图是 rgba，下面去除透明通道
    im = im[..., :3]
    # 需要复制，否则会映射到一个matlibplot的重用内存区，导致返回的图像会被破坏
    im = im.copy()
    return im


def draw_g(g: dgl.DGLHeteroGraph):
    # 把图结构画出来
    ng = g.to_networkx()
    pos = g.ndata['pos'].numpy().astype(np.float32)
    cls = g.ndata['pred_cls'].numpy().astype(np.int32)

    ax = plt.figure(111, (6., 6.), dpi=100, clear=True)
    # nx.draw(g, pos=dg.pts, node_size=50, node_color=[[.5, .5, .5, ]])
    nx.draw(ng, pos=pos, labels=dict(zip(np.arange(len(cls)), cls)), node_size=50, node_color=[[.1, 1., .1, ]])
    plt.xlabel('x')
    plt.ylabel('y')
    oim = tr_figure_to_array(ax)
    # plt.show()
    return oim


def main():
    # 图数据生成和读取器
    ds = DatasetReader()

    in_ch = 2
    n_class = 2
    batch_size = 10
    lr = 1e-3
    use_cuda = True

    # 是否对输入数据进行标准化，可以收敛更快
    use_input_norm = False

    # 总训练轮数
    n_epoch = 100
    # 训练时，循环多少次
    train_batch_count = 50

    train_show_dir = 'show_dir'
    os.makedirs(train_show_dir, exist_ok=True)

    if use_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # 创建图网络
    model = GNet(in_ch, n_class)
    model.to(device)

    # 损失函数
    loss_func = torch.nn.CrossEntropyLoss()

    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 开始训练
    for epoch in range(n_epoch):
        # 训练
        model.train()
        for batch_i, batch_g in enumerate(ds.train_batch_gen(batch_size, train_batch_count)):

            if use_input_norm:
                # 将输入数据标准化，可以收敛更快
                batch_g = norm_each_g_pos_feats(batch_g)

            # 把小图打包为一个大图
            batch_g = dgl.batch(batch_g)
            # 移动大图到指定设备上
            batch_g = batch_g.to(device)

            # 获得要训练的图特征和图标签
            batch_feats = batch_g.ndata['pos']
            batch_label = batch_g.ndata['cls']

            # 进行训练
            o = model(batch_g, batch_feats)
            loss = loss_func(o, batch_label)

            optim.zero_grad()
            loss.backward()
            optim.step()

        # 验证
        model.eval()
        # 节点结果，当一个节点分类正确时，该次节点结果记为1，否则记为0
        node_result = []
        # 每个图结果，当一个图的全部节点都分类正确时，该次图结果记为1，否则记为0
        graph_result = []

        oim_i = 0
        with torch.no_grad():
            for batch_i, batch_g in enumerate(ds.valid_batch_gen(batch_size)):

                if use_input_norm:
                    # 将输入数据标准化，可以收敛更快
                    batch_g = norm_each_g_pos_feats(batch_g)

                # 把多个图打包为一个大图
                batch_g = dgl.batch(batch_g).to(device)

                # 把图特征输入图网络进行预测
                batch_feats = batch_g.ndata['pos']
                o = model(batch_g, batch_feats)

                # 把预测结果存入图中，方面用于后面的图分割
                batch_g.ndata['pred_cls'] = torch.argmax(o, 1)

                # 把数据挪回CPU
                batch_g = batch_g.to('cpu')

                # 把大图再次分解为各个小图
                batch_g = dgl.unbatch(batch_g)

                # 循环每个小图，然后统计数据
                for g_i, g in enumerate(batch_g):
                    oim_i += 1
                    if epoch % 10 == 0 and oim_i <= 5:
                        oim = draw_g(g)
                        oim_path = f'{train_show_dir}/{epoch}_{oim_i}.jpg'
                        imageio.imwrite(oim_path, oim)

                    g_label = g.ndata['cls']
                    g_pred = g.ndata['pred_cls']
                    r = (g_label == g_pred).cpu().numpy()
                    node_result.extend(r)
                    graph_result.append(np.all(r))

        node_acc = np.mean(np.float32(node_result))
        graph_acc = np.mean(np.float32(graph_result))
        print(f"Epoch {epoch:05d} | VALID | Node Accuracy {node_acc:.4f} | Graph Accuracy {graph_acc:.4f}")

    # 开始测试
    model.eval()
    # 节点结果，当一个节点分类正确时，该次节点结果记为1，否则记为0
    node_result = []
    # 每个图结果，当一个图的全部节点都分类正确时，该次图结果记为1，否则记为0
    graph_result = []

    with torch.no_grad():
        for batch_i, batch_g in enumerate(ds.test_batch_gen(batch_size)):

            if use_input_norm:
                # 将输入数据标准化，可以收敛更快
                batch_g = norm_each_g_pos_feats(batch_g)

            # 把多个图打包为一个大图
            batch_g = dgl.batch(batch_g).to(device)

            # 把图特征输入图网络进行预测
            batch_feats = batch_g.ndata['pos']
            o = model(batch_g, batch_feats)

            # 把预测结果存入图中，方面用于后面的图分割
            batch_g.ndata['pred_cls'] = torch.argmax(o, 1)

            # 把数据挪回CPU
            batch_g = batch_g.to('cpu')

            # 把大图再次分解为各个小图
            batch_g = dgl.unbatch(batch_g)

            # 循环每个小图，然后统计数据
            for g in batch_g:
                g_label = g.ndata['cls']
                g_pred = g.ndata['pred_cls']
                r = (g_label == g_pred).cpu().numpy()
                node_result.extend(r)
                graph_result.append(np.all(r))

    node_acc = np.mean(np.float32(node_result))
    graph_acc = np.mean(np.float32(graph_result))
    print(f"Test | Node Accuracy {node_acc:.4f} | Graph Accuracy {graph_acc:.4f}")


if __name__ == '__main__':
    main()
