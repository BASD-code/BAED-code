# 生成子图embedding
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch_geometric.data import data
import torch_geometric as pyg
import random
from functools import partial
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GAE
import networkx as nx
import pickle as pkl
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
import pickle
import numpy as np
from torch_geometric.nn import GAE
import pandas as pd
import os
from dgl.data.utils import load_graphs, save_graphs
from torch_geometric.utils import subgraph
from encoder import GCNEncoder


def getNodeEmbedding(dataset, x, edge_index):
    # 加载权重获取模型
    if dataset == 'elliptic':
        model = torch.load("./weight/elliptic_gae_relu_256.pt")
        # 拼接特征至44417，166；x的特征为【n,166维度】，需要将其拼接为44417，166维度，用0进行padding
        padded_x = torch.zeros((44417, 166), device=x.device)
        padded_x[:x.size(0), :x.size(1)] = x
        # 将拼接好的向量输入model.encode
        Z = model.encode(padded_x, edge_index)
        # 复原x中当时位置的向量
        original_x = Z[:x.size(0), :x.size(1)]
        return original_x

# 示例调用
# if __name__ == "__main__":
#     # 随机生成一个[43, 166]维度的向量和一个[2, 17]维度的向量赋值给x和edge_index
#     x = torch.randn(43, 166)
#     edge_index = torch.randint(0, 43, (2, 17))
#     # 将x和edge_index移动到GPU（如果可用）
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     x = x.to(device)
#     edge_index = edge_index.to(device)
#     embeddings = getNodeEmbedding('elliptic', x, edge_index)
    """
    x: torch.Size([43, 166])
    edge_index: torch.Size([2, 17])
    Node Embeddings: torch.Size([43, 166])
    embedding: tensor([[-0.0200, -0.0069, -0.0199,  ..., -0.0756, -0.0345,  0.0581],
            [-0.0228, -0.0450, -0.0036,  ..., -0.0569, -0.0409,  0.0138],
            [ 0.0087, -0.0407,  0.0127,  ..., -0.0563, -0.0175,  0.0291],
            ...,
            [-0.0169, -0.0272, -0.0346,  ..., -0.0231, -0.0280,  0.0017],
            [-0.0316, -0.0372, -0.0041,  ..., -0.0311, -0.0161,  0.0192],
            [-0.0270, -0.0699, -0.0150,  ..., -0.0438, -0.0511,  0.0638]],
        device='cuda:0', grad_fn=<SliceBackward>)
    """
