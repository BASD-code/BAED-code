import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

#gae的encoder
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=False) # cached only for transductive learning
        # 设置激活函数
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError("Invalid activation function. Use 'relu', 'sigmoid', or 'tanh'.")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)  # 使用自定义激活函数
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)
        # 设置激活函数
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError("Invalid activation function. Use 'relu', 'sigmoid', or 'tanh'.")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)  # 使用自定义激活函数
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)



