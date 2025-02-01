# 保存用于训练gae的数据：分割好训练集和测试集的pkl文件
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
from encoder import GCNEncoder

import scipy.sparse as sp
import scipy.io as sio

import torch
import random
from collections import Counter
import dgl

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


# 将adj转换为dgl格式的图
def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph

# https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial6/Tutorial6.ipynb
# 将 NetworkX 图转换为 PyTorch Geometric Data 对象
def nx_to_pyg(nx_g):
    edge_index = torch.tensor(list(nx_g.edges)).t().contiguous()
    node_features = [nx_g.nodes[data]['feature'] for data in nx_g.nodes]
    node_features=torch.stack(node_features)
    node_labels=[nx_g.nodes[data]['label'] for data in nx_g.nodes]
    data = Data(x=node_features, edge_index=edge_index,y=node_labels)
    return data

def load_dataset():
    # 加载 .npz 文件中的数据
    graphItem = np.load("../graphs/dgraphfin.npz")
    x = torch.tensor(graphItem['x'])
    y = graphItem['y']
    y=torch.tensor(y)
    #进行转换，适配geometry
    edge_index = graphItem['edge_index']
    edge_index=torch.tensor(edge_index)
    data = Data(x=x, edge_index=edge_index.T,y=y)
    return data

def load_elliptic_data(data_dir, start_ts, end_ts):
    classes_csv = 'elliptic_txs_classes.csv'
    edgelist_csv = 'elliptic_txs_edgelist.csv'
    features_csv = 'elliptic_txs_features.csv'

    classes = pd.read_csv(os.path.join(data_dir, classes_csv), index_col='txId')  # labels for the transactions i.e. 'unknown', '1', '2'
    edgelist = pd.read_csv(os.path.join(data_dir, edgelist_csv), index_col='txId1')  # directed edges between transactions
    features = pd.read_csv(os.path.join(data_dir, features_csv), header=None, index_col=0)  # features of the transactions

    num_features = features.shape[1]
    num_tx = features.shape[0]
    total_tx = list(classes.index)

    # select only the transactions which are labelled
    labelled_classes = classes[classes['class'] != 'unknown']
    # classes['class'] = classes['class'].replace('unknown', '3')
    # labelled_classes=classes
    labelled_tx = list(labelled_classes.index)

    # to calculate a list of adjacency matrices for the different timesteps
    adj_mats = []
    features_labelled_ts = []
    classes_ts = []
    num_ts = 49  # number of timestamps from the paper

    for ts in range(start_ts, end_ts):
        features_ts = features[features[1] == ts + 1]
        tx_ts = list(features_ts.index)

        labelled_tx_ts = [tx for tx in tx_ts if tx in set(labelled_tx)]

        # adjacency matrix for all the transactions
        # we will only fill in the transactions of this timestep which have labels and can be used for training
        adj_mat = pd.DataFrame(np.zeros((num_tx, num_tx)), index=total_tx, columns=total_tx)

        edgelist_labelled_ts = edgelist.loc[edgelist.index.intersection(labelled_tx_ts).unique()]
        for i in range(edgelist_labelled_ts.shape[0]):
            adj_mat.loc[edgelist_labelled_ts.index[i], edgelist_labelled_ts.iloc[i]['txId2']] = 1

        adj_mat_ts = adj_mat.loc[labelled_tx_ts, labelled_tx_ts]
        features_l_ts = features.loc[labelled_tx_ts]

        adj_mats.append(adj_mat_ts)
        features_labelled_ts.append(features_l_ts)
        classes_ts.append(classes.loc[labelled_tx_ts])

    return adj_mats, features_labelled_ts, classes_ts


def load_random_dataset(dataset):
    if dataset=="dgraph":
        graphItem = np.load("../graphs/dgraphfin.npz")
        # 将 x 转换为 PyTorch 张量
        x = torch.tensor(graphItem['x'], dtype=torch.float)
        # 将 y 转换为 PyTorch 张量
        y = torch.tensor(graphItem['y'], dtype=torch.long)
        # 将 edge_index 转换为 PyTorch 张量，并确保它是一个包含两个张量的元组
        edge_index_np = graphItem['edge_index']
        edge_index = torch.tensor(edge_index_np, dtype=torch.long).T
        # 确定采样的节点数量
        num_nodes = x.size(0)
        num_sampled_nodes = int(num_nodes * 0.05)
        # 随机采样节点
        # sampled_node_indices = np.random.choice(num_nodes, num_sampled_nodes, replace=False)
        sampled_node_indices=np.array([i for i in range(num_sampled_nodes)])
        sampled_node_indices_tensor = torch.tensor(sampled_node_indices, dtype=torch.long)
        # 更新 x 和 y
        x_sampled = x[sampled_node_indices]
        y_sampled = y[sampled_node_indices]
        # 更新 edge_index，只保留与采样节点相关的边
        row,col=edge_index[0],edge_index[1]
        # 保留源节点和目标节点都在采样节点集合中的边
        mask = (torch.tensor([i in sampled_node_indices_tensor for i in row]) & torch.tensor([i in sampled_node_indices_tensor for i in col]))
        edge_index_sampled = edge_index[:, mask]
        # 创建子数据集
        data = Data(x=x_sampled, edge_index=edge_index_sampled, y=y_sampled)
    elif dataset=="elliptic":
        data_dir = "/home1/hsy/graph/GAE/originDataset/elliptic_bitcoin_dataset"
        adj_mats, features_labelled_ts, classes_ts = load_elliptic_data(data_dir, 1, 49)
        # 获取t=0阶段的特征训练GAE
        # Convert adj_mats to edge_index tensor
        edge_index_list = []
        for adj_mat in adj_mats:
            src, dst = np.nonzero(adj_mat.values)
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_index_list.append(edge_index)
        edge_index = torch.cat(edge_index_list, dim=1)
        # Convert features_labelled_ts to node_features tensor
        node_features = torch.tensor(pd.concat(features_labelled_ts).values, dtype=torch.float32)
        # Convert classes_ts to node_labels tensor
        labels = pd.concat(classes_ts)
        node_labels = torch.tensor(labels['class'].apply(lambda x: 1 if x == '1' else 0).values, dtype=torch.int64)
        # 获取label
        data = Data(x=node_features, edge_index=edge_index, y=node_labels)
    elif dataset=='tfinance':
        graph, label_dict = load_graphs('/home1/hsy/graph/GAE/originDataset/tfinance')
        graph = graph[0]
        labels = graph.ndata['label'].argmax(1).to(torch.int64)  # 确保标签是长整型
        features = graph.ndata['feature'].to(torch.float32)  # 确保特征是浮点型
        # Select the first 30000 nodes
        num_nodes = 10000
        selected_nodes = torch.arange(num_nodes)
        # Extract the features and labels for the selected nodes
        selected_features = features[selected_nodes]
        selected_labels = labels[selected_nodes]
        # Get the edges and update edge_index for the selected nodes
        src, dst = graph.edges()
        edge_index = torch.stack([src, dst], dim=0)
        # Create a subgraph with the selected nodes
        sub_edge_index, _ = subgraph(selected_nodes, edge_index, relabel_nodes=True)
        # Create Data object
        data = Data(x=selected_features, edge_index=sub_edge_index, y=selected_labels)
    elif dataset=="reddit":
        data = sio.loadmat("../graphs/{}.mat".format(dataset))
        label = data['Label'] if ('Label' in data) else data['gnd']
        attr = data['Attributes'] if ('Attributes' in data) else data['X']
        network = data['Network'] if ('Network' in data) else data['A']
        adj = sp.csr_matrix(network)
        feat = sp.lil_matrix(attr)
        ano_labels = np.squeeze(np.array(label))
        if 'str_anomaly_label' in data:
            str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
            attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
        else:
            str_ano_labels = None
            attr_ano_labels = None
        num_node = adj.shape[0]
        all_idx = list(range(num_node))
        # Sample some labeled normal nodes
        all_normal_label_idx = [i for i in all_idx if ano_labels[i] == 0]
        # abnormal index
        all_abnormal_label_idx = [i for i in all_idx if ano_labels[i] == 1]
        # features = feat.todense()
        features = preprocess_features(feat)
        graph = adj_to_dgl_graph(adj)
        nb_nodes = feat.shape[0]
        ft_size = feat.shape[1]
        graph.ndata["feature"]=torch.tensor(features,dtype=torch.float32)
        graph.ndata["label"]=torch.tensor(ano_labels,dtype=torch.int64)
        labels = graph.ndata['label']
        features = graph.ndata['feature']
        # 将 DGL 图转换为 NetworkX 图
        nx_graph = dgl.to_networkx(graph, node_attrs=["feature", "label"])
        data=nx_graph.edges()
        # 拆分数据
        src = [x[0] for x in data]
        dst = [x[1] for x in data]
        # 转换为tensor
        edge_index = torch.tensor([src,dst])
        data = Data(x=graph.ndata["feature"], edge_index=edge_index, y=graph.ndata["label"])
    elif dataset=="photo":
        data = sio.loadmat("../graphs/{}.mat".format(dataset))
        label = data['Label'] if ('Label' in data) else data['gnd']
        attr = data['Attributes'] if ('Attributes' in data) else data['X']
        network = data['Network'] if ('Network' in data) else data['A']
        adj = sp.csr_matrix(network)
        feat = sp.lil_matrix(attr)
        ano_labels = np.squeeze(np.array(label))
        if 'str_anomaly_label' in data:
            str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
            attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
        else:
            str_ano_labels = None
            attr_ano_labels = None
        num_node = adj.shape[0]
        all_idx = list(range(num_node))
        # Sample some labeled normal nodes
        all_normal_label_idx = [i for i in all_idx if ano_labels[i] == 0]
        # abnormal index
        all_abnormal_label_idx = [i for i in all_idx if ano_labels[i] == 1]
        features = feat.todense()
        # features = preprocess_features(feat)
        graph = adj_to_dgl_graph(adj)
        nb_nodes = feat.shape[0]
        ft_size = feat.shape[1]
        graph.ndata["feature"]=torch.tensor(features,dtype=torch.float32)
        graph.ndata["label"]=torch.tensor(ano_labels,dtype=torch.int64)
        labels = graph.ndata['label']
        features = graph.ndata['feature']
        # 将 DGL 图转换为 NetworkX 图
        nx_graph = dgl.to_networkx(graph, node_attrs=["feature", "label"])
        data=nx_graph.edges()
        # 拆分数据
        src = [x[0] for x in data]
        dst = [x[1] for x in data]
        # 转换为tensor
        edge_index = torch.tensor([src,dst])
        data = Data(x=graph.ndata["feature"], edge_index=edge_index, y=graph.ndata["label"])
    elif dataset=="Amazon":
        data = sio.loadmat("../graphs/{}.mat".format(dataset))
        label = data['Label'] if ('Label' in data) else data['gnd']
        attr = data['Attributes'] if ('Attributes' in data) else data['X']
        network = data['Network'] if ('Network' in data) else data['A']
        adj = sp.csr_matrix(network)
        feat = sp.lil_matrix(attr)
        ano_labels = np.squeeze(np.array(label))
        if 'str_anomaly_label' in data:
            str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
            attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
        else:
            str_ano_labels = None
            attr_ano_labels = None
        num_node = adj.shape[0]
        all_idx = list(range(num_node))
        # Sample some labeled normal nodes
        all_normal_label_idx = [i for i in all_idx if ano_labels[i] == 0]
        # abnormal index
        all_abnormal_label_idx = [i for i in all_idx if ano_labels[i] == 1]
        # features = feat.todense()
        features = preprocess_features(feat)
        graph = adj_to_dgl_graph(adj)
        nb_nodes = feat.shape[0]
        ft_size = feat.shape[1]
        graph.ndata["feature"]=torch.tensor(features,dtype=torch.float32)
        graph.ndata["label"]=torch.tensor(ano_labels,dtype=torch.int64)
        labels = graph.ndata['label']
        features = graph.ndata['feature']
        # 将 DGL 图转换为 NetworkX 图
        nx_graph = dgl.to_networkx(graph, node_attrs=["feature", "label"])
        data=nx_graph.edges()
        # 拆分数据
        src = [x[0] for x in data]
        dst = [x[1] for x in data]
        # 转换为tensor
        edge_index = torch.tensor([src,dst])
        data = Data(x=graph.ndata["feature"], edge_index=edge_index, y=graph.ndata["label"])
    return data   



# data: Data(x=[3700550, 17], edge_index=[2, 4300999], y=[3700550])
# elliptic:data: Data(x=[44417, 166], edge_index=[2, 34700], y=[44417])
# reddit: Data(x=[10984, 64], edge_index=[2, 168016], y=[10984])
# photo:data: Data(x=[7535, 745], edge_index=[2, 238163], y=[7535])
# Amazon:data: Data(x=[11944, 25], edge_index=[2, 8796784], y=[11944])

# dgraph数据集过大，在计算邻接矩阵时会报错，因此随机采样10000个节点作为训练样本进行训练
print("dgraph train dataset save")
data=load_random_dataset("dgraph")
print("data:",data)
data = train_test_split_edges(data)
print("data:",data)
torch.save(data, "./data/dgraph_gae.pt")