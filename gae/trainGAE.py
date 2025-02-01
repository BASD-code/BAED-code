# 通过torch_geometry训练GAE
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch_geometric.data import data
import torch_geometric as pyg
import random
from functools import partial
import pandas as pd
#读取数据集
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GAE
import networkx as nx
import pickle as pkl
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
import pickle
import numpy as np
from torch_geometric.nn import GAE
import pandas as pd
import os
from dgl.data.utils import load_graphs, save_graphs
from torch_geometric.utils import subgraph
from encoder import GCNEncoder,VariationalGCNEncoder
from torch_geometric.nn import VGAE



def load_data(filepath):
    data = torch.load(filepath)
    print("data:", data)
    return data

def configure_model(data, out_channels=64, num_features=17,activation='relu',gae_type="GAE"):
    if gae_type=="gae":
        model = GAE(GCNEncoder(num_features, out_channels, activation=activation))
    elif gae_type=="VGAE":
        model = VGAE(VariationalGCNEncoder(num_features, out_channels, activation=activation))  
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    model = model.to(device)
    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    return model, device, x, train_pos_edge_index, optimizer

def train(model, optimizer, x, train_pos_edge_index):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)

def test(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)



def main():
    dataset="dgraph"
    filepath=f"./data/{dataset}_gae.pt"
    data = load_data(filepath)
    # 配置参数
    """
    elliptic:
    out_channels = 256
    num_features = 166
    """
    out_channels = 32
    num_features =17
    gae_type="gae"
    activation= 'sigmoid'
    epochs = 100
    model, device, x, train_pos_edge_index, optimizer = configure_model(data, out_channels, num_features,activation,gae_type)
    print("model:",model)
    # best_auc = 0.0
    # best_model = None
    # for epoch in range(1, epochs + 1):
    #     loss = train(model, optimizer, x, train_pos_edge_index)
    #     auc, ap = test(model, x, train_pos_edge_index, data.test_pos_edge_index, data.test_neg_edge_index)
    #     print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    #     if auc > best_auc:
    #         best_auc = auc
    #         best_model = model

    # print("Best AUC:", best_auc)
    # torch.save(best_model, f'./weight/{dataset}_{gae_type}_{activation}_{out_channels}.pt')
    # print("x:", x.size())
    # print("train_pos_edge_index:", train_pos_edge_index.size())
    # Z = best_model.encode(x, train_pos_edge_index)
    # print("Z:", Z.size())

if __name__ == "__main__":
    main()