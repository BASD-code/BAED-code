# 生成子图数据集
# 生成方法：以正常节点或者异常节点作为核心节点，计算二跳子图
# 数据集选择：5个基准数据集 dgraph，tfinance，elliptic,xxxx
import math
import torch
import os 
import networkx as nx
import numpy as np
import dgl
from dgl.data.utils import load_graphs, save_graphs
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch_geometric.data import data
import torch_geometric as pyg
import random
from functools import partial
from torch_geometric.datasets import QM9
from sklearn.model_selection import train_test_split
import pandas as pd
import torch_geometric.data
import scipy.sparse as sp
import scipy.io as sio

import torch
import random
from collections import Counter


import threading
import time
# 定义一个函数包装器，用于检测函数的执行时间
class TimeoutWrapper:
    def __init__(self, func, timeout):
        self.func = func
        self.timeout = timeout
        self.result = None

    def _run_func(self, *args, **kwargs):
        try:
            self.result = self.func(*args, **kwargs)
        except Exception as e:
            self.result = e

    def __call__(self, *args, **kwargs):
        thread = threading.Thread(target=self._run_func, args=args, kwargs=kwargs)
        thread.start()
        thread.join(self.timeout)  # 等待线程完成，最长时间为 self.timeout 秒

        if thread.is_alive():
            # 如果线程仍然在运行，则超时
            return 'timeout'
        return self.result

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

#处理elliptic数据集三个文件的基本信息
def load_data(data_dir, start_ts, end_ts):
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

# 将adj转换为dgl格式的图
def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph



# 生成两种类型的子图数据集：label为1的子图数据集用于训练数据生成，label为1和0的子图数据集用于实现子图异常检测
def initOriginDataset(dataset, label_type):
    print("label_type:",label_type)
    print("dataset:",dataset)
    if dataset == 'dgraph':
        graphItem = np.load("../graphs/dgraphfin.npz")
        x = graphItem['x']
        y = graphItem['y']
        edge_index = graphItem['edge_index']
        # 提取源节点和目标节点
        src = torch.tensor(edge_index[:, 0], dtype=torch.int64)
        dst = torch.tensor(edge_index[:, 1], dtype=torch.int64)
        # 创建 DGL 图
        graph = dgl.graph((src, dst))
        # 添加节点特征和标签
        graph.ndata['feature'] = torch.tensor(x, dtype=torch.float32)
        graph.ndata['label'] = torch.tensor(y, dtype=torch.int64)
        labels = graph.ndata['label']
        features = graph.ndata['feature']
        # 将 DGL 图转换为 NetworkX 图
        nx_graph = dgl.to_networkx(graph, node_attrs=["feature", "label"])
        # 保存dgraph子图数据集
        if label_type == 'label_1':
            # 获取标签为1的节点
            target_nodes = torch.nonzero(labels == 1, as_tuple=True)[0].tolist()
            file_name = f'../graphs/{dataset}_init_1_dataset.pkl'
        elif label_type == 'label_0_1':
            # 获取标签为1或者0的节点
            target_nodes = torch.nonzero((labels == 1) | (labels == 0), as_tuple=True)[0].tolist()
            file_name = f'../graphs/{dataset}_init_0_1_dataset.pkl'
        else:
            raise ValueError("Invalid label_type. Use 'label_1' or 'label_0_1'.")
        # 构建子图列表
        nx_graphs = []
        for node in target_nodes:
            # 获取2跳节点
            two_hop_nodes = set(nx.single_source_shortest_path_length(nx_graph, node, cutoff=2).keys())
            # 构建子图
            subgraph = nx_graph.subgraph(two_hop_nodes).copy()
            # 将 MultiDiGraph 转换为 Graph
            subgraph = nx.Graph(subgraph)
            # 增加subgraph的label
            subgraph.graph['sublabel'] = int(labels[node].item())
            # 添加到子图列表
            nx_graphs.append(subgraph)
        # 保存
        print("正在保存数据文件：",file_name)
        with open(file_name, 'wb') as f:
            pkl.dump(nx_graphs, f)
    elif dataset == 'elliptic':
        # 加载数据
        data_dir = "../graphs/elliptic_bitcoin_dataset"
        adj_mats, features_labelled_ts, classes_ts = load_data(data_dir, 1, 49)
        # 创建 DGL 图
        graphs = []
        for i in range(len(adj_mats)):
            adj_mat = adj_mats[i]
            features = features_labelled_ts[i]
            labels = classes_ts[i]
            # 提取边列表
            src, dst = np.nonzero(adj_mat.values)
            src = torch.tensor(src, dtype=torch.int64)
            dst = torch.tensor(dst, dtype=torch.int64)
            # 创建 DGL 图
            graph = dgl.graph((src, dst))
            # 确保特征和标签的数量与图中的节点数量匹配
            node_features = torch.tensor(features.values, dtype=torch.float32)
            # 1不合法，0合法
            node_labels = torch.tensor(labels['class'].apply(lambda x: 1 if x == '1' else 0).values, dtype=torch.int64)
            if len(node_features) == graph.num_nodes() and len(node_labels) == graph.num_nodes():
                graph.ndata['feature'] = node_features
                graph.ndata['label'] = node_labels
                graphs.append(graph)
            else:
                print(f"Skipping graph {i} due to mismatch in number of nodes and features/labels")
        # 打印图信息
        nx_graphs = []
        for g in graphs:
            nx_graph = dgl.to_networkx(g, node_attrs=["feature", "label"])
            labels = g.ndata['label']
            features = g.ndata['feature']
            if label_type == 'label_1':
                # 获取标签为1的节点
                target_nodes = torch.nonzero(labels == 1, as_tuple=True)[0].tolist()
                file_name = '../graphs/elliptic_init_1_dataset.pkl'
            elif label_type == 'label_0_1':
                # 获取标签为1或者0的节点
                target_nodes = torch.nonzero((labels == 1) | (labels == 0), as_tuple=True)[0].tolist()
                file_name = '../graphs/elliptic_init_0_1_dataset.pkl'
            else:
                raise ValueError("Invalid label_type. Use 'label_1' or 'label_0_1'.")
            for node in target_nodes:
                # 获取2跳节点
                two_hop_nodes = set(nx.single_source_shortest_path_length(nx_graph, node, cutoff=2).keys())
                # 构建子图
                subgraph = nx_graph.subgraph(two_hop_nodes).copy()
                # 将 MultiDiGraph 转换为 Graph
                subgraph = nx.Graph(subgraph)
                subgraph.graph['sublabel'] = int(labels[node].item())
                # 添加到子图列表
                nx_graphs.append(subgraph)
        # 保存
        with open(file_name, 'wb') as f:
            pkl.dump(nx_graphs, f)
    elif dataset=='tfinance':
       # 加载图数据
        graph, label_dict = load_graphs('../graphs/tfinance')
        graph = graph[0]
        labels = graph.ndata['label'].argmax(1)
        features = graph.ndata['feature']
        # 将 DGL 图转换为 NetworkX 图
        nx_graph = dgl.to_networkx(graph, node_attrs=["feature", "label"])
        if label_type == 'label_1':
            # 获取标签为1的节点
            target_nodes = torch.nonzero(labels == 1, as_tuple=True)[0].tolist()
            n=len(target_nodes)
            target_nodes=target_nodes[:200]
            file_name = '../graphs/tfinance_init_1_dataset.pkl'
        elif label_type == 'label_0_1':
            # 获取标签为1或者0的节点
            # target_nodes = torch.nonzero((labels == 1) | (labels == 0), as_tuple=True)[0].tolist()
            target_nodes_0 = torch.nonzero((labels == 0), as_tuple=True)[0].tolist()
            target_nodes_1 = torch.nonzero((labels == 1), as_tuple=True)[0].tolist()
            # 抽取 0 类的 0.05%
            num_samples_0 = max(1, int(0.05 * len(target_nodes_0)))  # 确保至少抽取 1 个
            sampled_nodes_0 = random.sample(target_nodes_0, num_samples_0)
            # 抽取 1 类的 50%
            num_samples_1 = max(1, int(0.25 * len(target_nodes_1)))  # 确保至少抽取 1 个
            sampled_nodes_1 = random.sample(target_nodes_1, num_samples_1)
            print("0:",len(sampled_nodes_0))
            print("1:",len(sampled_nodes_1))
            # 合并抽取的节点列表
            target_nodes = sampled_nodes_0 + sampled_nodes_1
            file_name = '../graphs/tfinance_init_0_1_dataset.pkl'
        else:
            raise ValueError("Invalid label_type. Use 'label_1' or 'label_0_1'.")
        # 构建子图列表
        nx_graphs = []
        for node in target_nodes:
            try:
                # 获取2跳节点
                two_hop_nodes = set(nx.single_source_shortest_path_length(nx_graph, node, cutoff=1).keys())
                print("node:",node,len(two_hop_nodes))
                if len(two_hop_nodes)>1200:
                    print("edge index过大")
                    continue
                # 构建子图
                subgraph = nx_graph.subgraph(two_hop_nodes).copy()
                # 将 MultiDiGraph 转换为 Graph
                subgraph = nx.Graph(subgraph)
                subgraph.graph['sublabel'] = int(labels[node].item())
                # 添加到子图列表
                nx_graphs.append(subgraph)
            except Exception as e:
                print(f"节点 {node} 存在错误，跳过该节点: {e}")
        # 保存
        with open(file_name, 'wb') as f:
            pkl.dump(nx_graphs, f)
    elif dataset=='photo':
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
        print("normal:",len(all_normal_label_idx))
        # abnormal index
        all_abnormal_label_idx = [i for i in all_idx if ano_labels[i] == 1]
        print("abnorm:",len(all_abnormal_label_idx))
        features = feat.todense()
        graph = adj_to_dgl_graph(adj)
        nb_nodes = feat.shape[0]
        ft_size = feat.shape[1]
        # print("dgl_graph:",dgl_graph)
        # print("features:",torch.tensor(features).shape)
        # print("label:",torch.tensor(ano_labels))
        # print("nb_nodes:",nb_nodes)
        # print("ft_size:",ft_size)
        graph.ndata["feature"]=torch.tensor(features,dtype=torch.float32)
        graph.ndata["label"]=torch.tensor(ano_labels,dtype=torch.int64)
        labels = graph.ndata['label']
        features = graph.ndata['feature']
        # 将 DGL 图转换为 NetworkX 图
        nx_graph = dgl.to_networkx(graph, node_attrs=["feature", "label"])
        # print("nx_graph:",nx_graph.nodes(data=True))
        # 保存dgraph子图数据集
        if label_type == 'label_1':
            # 获取标签为1的节点
            target_nodes = torch.nonzero(labels == 1, as_tuple=True)[0].tolist()
            file_name = f'../graphs/{dataset}_init_1_dataset.pkl'
        elif label_type == 'label_0_1':
            # 获取标签为1或者0的节点
            target_nodes = torch.nonzero((labels == 1) | (labels == 0), as_tuple=True)[0].tolist()
            file_name = f'../graphs/{dataset}_init_0_1_dataset.pkl'
        else:
            raise ValueError("Invalid label_type. Use 'label_1' or 'label_0_1'.")
        # 构建子图列表
        nx_graphs = []
        # print("len(target_nodes):",target_nodes)
        for node in target_nodes:
            # 获取2跳节点
            two_hop_nodes = set(nx.single_source_shortest_path_length(nx_graph, node, cutoff=1).keys())
            # 构建子图
            subgraph = nx_graph.subgraph(two_hop_nodes).copy()
            # 将 MultiDiGraph 转换为 Graph
            subgraph = nx.Graph(subgraph)
            # 增加subgraph的label
            subgraph.graph['sublabel'] = int(labels[node].item())
            # 添加到子图列表
            nx_graphs.append(subgraph)
        # 保存
        print("正在保存数据文件：",file_name)
        with open(file_name, 'wb') as f:
            pkl.dump(nx_graphs, f)
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
        print("正常节点数目:",len(all_normal_label_idx))
        # abnormal index
        all_abnormal_label_idx = [i for i in all_idx if ano_labels[i] == 1]
        print("异常节点数目:",len(all_abnormal_label_idx))
        # features = feat.todense()
        features = preprocess_features(feat)
        graph = adj_to_dgl_graph(adj)
        nb_nodes = feat.shape[0]
        ft_size = feat.shape[1]
        # print("dgl_graph:",dgl_graph)
        # print("features:",torch.tensor(features).shape)
        # print("label:",torch.tensor(ano_labels))
        # print("nb_nodes:",nb_nodes)
        # print("ft_size:",ft_size)
        graph.ndata["feature"]=torch.tensor(features,dtype=torch.float32)
        graph.ndata["label"]=torch.tensor(ano_labels,dtype=torch.int64)
        labels = graph.ndata['label']
        features = graph.ndata['feature']
        # 将 DGL 图转换为 NetworkX 图
        nx_graph = dgl.to_networkx(graph, node_attrs=["feature", "label"])
        # print("nx_graph:",nx_graph.nodes(data=True))
        # 保存dgraph子图数据集
        if label_type == 'label_1':
            # 获取标签为1的节点
            target_nodes = torch.nonzero(labels == 1, as_tuple=True)[0].tolist()
            file_name = f'../graphs/{dataset}_init_1_dataset.pkl'
        elif label_type == 'label_0_1':
            # 获取标签为1或者0的节点
            target_nodes = torch.nonzero((labels == 1) | (labels == 0), as_tuple=True)[0].tolist()
            file_name = f'../graphs/{dataset}_init_0_1_dataset.pkl'
        else:
            raise ValueError("Invalid label_type. Use 'label_1' or 'label_0_1'.")
        # 构建子图列表
        nx_graphs = []
        for node in target_nodes:
            # 获取2跳节点
            two_hop_nodes = set(nx.single_source_shortest_path_length(nx_graph, node, cutoff=1).keys())
            # 构建子图
            subgraph = nx_graph.subgraph(two_hop_nodes).copy()
            # 将 MultiDiGraph 转换为 Graph
            subgraph = nx.Graph(subgraph)
            # 增加subgraph的label
            subgraph.graph['sublabel'] = int(labels[node].item())
            # 添加到子图列表
            nx_graphs.append(subgraph)
        # 保存
        print("正在保存数据文件：",file_name)
        with open(file_name, 'wb') as f:
            pkl.dump(nx_graphs, f)
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
        print("正常节点数目:",len(all_normal_label_idx))
        # abnormal index
        all_abnormal_label_idx = [i for i in all_idx if ano_labels[i] == 1]
        print("异常节点数目:",len(all_abnormal_label_idx))
        # features = feat.todense()
        features = preprocess_features(feat)
        graph = adj_to_dgl_graph(adj)
        nb_nodes = feat.shape[0]
        ft_size = feat.shape[1]
        # print("dgl_graph:",dgl_graph)
        # print("features:",torch.tensor(features).shape)
        # print("label:",torch.tensor(ano_labels))
        # print("nb_nodes:",nb_nodes)
        # print("ft_size:",ft_size)
        graph.ndata["feature"]=torch.tensor(features,dtype=torch.float32)
        graph.ndata["label"]=torch.tensor(ano_labels,dtype=torch.int64)
        labels = graph.ndata['label']
        features = graph.ndata['feature']
        # 将 DGL 图转换为 NetworkX 图
        nx_graph = dgl.to_networkx(graph, node_attrs=["feature", "label"])
        # print("nx_graph:",nx_graph.nodes(data=True))
        # 保存dgraph子图数据集
        if label_type == 'label_1':
            # 获取标签为1的节点
            target_nodes = torch.nonzero(labels == 1, as_tuple=True)[0].tolist()
            file_name = f'../graphs/{dataset}_init_1_dataset.pkl'
        elif label_type == 'label_0_1':
            # 获取标签为1或者0的节点
            target_nodes = torch.nonzero((labels == 1) | (labels == 0), as_tuple=True)[0].tolist()
            file_name = f'../graphs/{dataset}_init_0_1_dataset.pkl'
        else:
            raise ValueError("Invalid label_type. Use 'label_1' or 'label_0_1'.")
        # 构建子图列表
        nx_graphs = []
        print("len(target_nodes):",len(target_nodes))
        index=len(target_nodes)
        # 包装 nx.single_source_shortest_path_length 函数
        # get_two_hop_nodes = TimeoutWrapper(nx.single_source_shortest_path_length, timeout=5)
        for node in target_nodes:
            print("index:",index)
            index-=1
            # 获取2跳节点
            two_hop_nodes = set(nx.single_source_shortest_path_length(nx_graph, node, cutoff=1).keys())
            # 构建子图
            subgraph = nx_graph.subgraph(two_hop_nodes).copy()
            # 将 MultiDiGraph 转换为 Graph
            subgraph = nx.Graph(subgraph)
            # 增加subgraph的label
            subgraph.graph['sublabel'] = int(labels[node].item())
            # 添加到子图列表
            nx_graphs.append(subgraph)
            if index==8000:
                print("正在保存数据文件：",file_name,index)
                with open(file_name, 'wb') as f:
                    pkl.dump(nx_graphs, f)
            elif index==6000:
                print("正在保存数据文件：",file_name,index)
                with open(file_name, 'wb') as f:
                    pkl.dump(nx_graphs, f)
            elif index==4000:
                print("正在保存数据文件：",file_name,index)
                with open(file_name, 'wb') as f:
                    pkl.dump(nx_graphs, f)
        # 保存
        print("正在保存数据文件：",file_name)
        with open(file_name, 'wb') as f:
            pkl.dump(nx_graphs, f)
#elliptic
# initOriginDataset('elliptic', 'label_1')
# initOriginDataset('elliptic', 'label_0_1')
# #dgraph
initOriginDataset('dgraph', 'label_1')
initOriginDataset('dgraph', 'label_0_1')
# #tfinance
# initOriginDataset('tfinance', 'label_1')
# initOriginDataset('tfinance', 'label_0_1')

# initOriginDataset('photo', 'label_1')
# initOriginDataset('photo', 'label_0_1')

# initOriginDataset('reddit', 'label_1')
# initOriginDataset('reddit', 'label_0_1')

# initOriginDataset('Amazon', 'label_1')
# initOriginDataset('Amazon', 'label_0_1')