# DGL-Classify-Ring
This is an example of DGL graph neural library.  
Graph neural network is used to classify whether the nodes of a fully connected graph are located in the inner ring or the outer ring.  
这是一个DGL图神经库的入门例子。  
使用图神经网络来分类全连接图的节点是位于内环还是外环。    

![1](./image/1.jpg)  
![2](./image/2.jpg)  
![3](./image/3.jpg)  


# Dependency / 依赖库
```txt
numpy >= 1.20
networkx >= 2.5.1
imageio >= 2.9.0
matplotlib >= 3.4.2
torch >= 1.8.1
dgl >= 0.6.1
```

# Start / 开始
非常简单，只需要克隆本仓库，使用以下命令启动即可。  
内外环图数据集会自动随机生成。  
网络会自动开始训练和验证。  
你可以在 show_dir 中看到训练中生成的图像。  

Very simple, just clone this repo and start it with the following command.
The inner and outer ring graph dataset is generated automatically and randomly.
The network will start training and verification automatically.
You can see the images generated during training in the folder "train_show_dir".

```bash
git clone this
cd ./DGL-Classify-Ring
python gnn_example.py
```
