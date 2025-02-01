import math
import numpy as np
import torch_geometric as pyg
import torch
import torch_scatter
from diffusion.diffusion_base import index_to_log_onehot
from torch.nn import functional as F
from torch import nn
from torch.nn.parameter import Parameter
# 引入计算子图embedding的函数
from gae.encoder import GCNEncoder
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GAE

norm_dict = {
    'Batch': lambda d: torch.nn.BatchNorm1d(d),
    'None': lambda d: torch.nn.Identity(),
    "Inst": lambda d: pyg.nn.norm.InstanceNorm(d),
    "Graph": lambda d: pyg.nn.norm.GraphNorm(d),
}


class SelEmb(torch.nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.act = act
        self.linear = torch.nn.Linear(in_dim, out_dim)
    def forward(self, t):
        out = self.act(t)
        out = self.linear(out)
        return out


class TimeEmb(torch.nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.act = act
        self.linear = torch.nn.Linear(in_dim, out_dim)
    def forward(self, t):
        out = self.act(t)
        out = self.linear(out)
        return out
        

class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class NodeModel(nn.Module):
    def __init__(self, num_bits, max_num_nodes, seq_lens, n_layers=6):
        super().__init__()
        self.num_bits = num_bits
        self.max_num_nodes = max_num_nodes
        self.seq_lens = seq_lens
        self.n_layers = n_layers
        self.embedding = nn.Linear(num_bits, 64)
        self.pos_embedding = SinusoidalPosEmb(64, seq_lens)
        self.g_embedding = nn.Linear(1, 256)
        self.res_embedding = nn.Linear(1, 64)
        self.lstm = nn.LSTM(input_size=64*3, hidden_size=256, num_layers=self.n_layers, batch_first=True)
        self.dropout = nn.Dropout(0)
        self.linear = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 128))
    def forward(self, x, g_v, res_count):
        x = self.embedding(x)
        # 归一化转为真正的分布
        g = g_v / self.max_num_nodes
        r = res_count / self.max_num_nodes

        # 网络映射为隐向量
        g = self.g_embedding(g)
        r = self.res_embedding(r[..., None])
        # 创建对应于X每一个位置的位置编码
        t = torch.arange(0,x.shape[1])[None,:].repeat_interleave(x.shape[0], 0).to(x.device).view(-1)
        t = self.pos_embedding(t).view(x.shape[0],-1, 64)
        x = torch.cat([x, t, r], dim=-1)
        g = g[None,:,:].repeat_interleave(self.n_layers,0)
        x, _ = self.lstm(x, (g, g))
        x = self.linear(self.dropout(x))
        return x

class BitModel(nn.Module):
    def __init__(self, num_bits, max_num_nodes, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(3, 64)
        self.max_num_nodes = max_num_nodes
        self.n_layers = n_layers
        self.num_bits = num_bits
        self.pos_embedding = SinusoidalPosEmb(64, num_bits)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=self.n_layers, batch_first=True)
        self.dropout = nn.Dropout(0)
        self.linear = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
        self.res_embedding = nn.Linear(1, 128) 

    def forward(self, bits, hidden_nodes, res_count):
        x = self.embedding(bits)
        r = res_count/self.max_num_nodes
        r = self.res_embedding(r)
        t = torch.arange(0,x.shape[1])[None, :].repeat_interleave(x.shape[0], 0).to(x.device).view(-1)
        t = self.pos_embedding(t).view(x.shape[0], -1, 64)
        x = torch.cat([x, t], dim=-1)
        hidden_nodes = torch.cat([hidden_nodes, r],dim=-1)
        hidden_nodes = hidden_nodes[None,...].repeat_interleave(self.n_layers,0)
        x, _ = self.lstm(x, (hidden_nodes, hidden_nodes))
        x = self.linear(self.dropout(x))
        return x

class MiniAttentionLayer(torch.nn.Module):
    def __init__(self, node_dim, in_edge_dim, out_edge_dim, d_model, num_heads=2):
        super().__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(d_model*num_heads, num_heads)
        self.qkv_node = torch.nn.Linear(node_dim, d_model * 3 * num_heads)
        self.qkv_edge = torch.nn.Linear(in_edge_dim, d_model * 3 * num_heads)
        self.edge_linear = torch.nn.Sequential(torch.nn.Linear(d_model * num_heads, d_model), 
                                                torch.nn.SiLU(), 
                                                torch.nn.Linear(d_model, out_edge_dim))
    def forward(self, node_us, node_vs, edges):

        # node_us/vs: (B, D)
        q_node_us, k_node_us, v_node_us = self.qkv_node(node_us).chunk(3, -1) # (B, D*num_heads) for q/k/v
        q_node_vs, k_node_vs, v_node_vs = self.qkv_node(node_vs).chunk(3, -1) # (B, D*num_heads) for q/k/v
        q_edges, k_edges, v_edges = self.qkv_edge(edges).chunk(3, -1) # (B, D*num_heads) for q/k/v

        q = torch.stack([q_node_us, q_node_vs, q_edges], 1) # (B, 3, D*num_heads)
        k = torch.stack([k_node_us, k_node_vs, k_edges], 1) # (B, 3, D*num_heads)
        v = torch.stack([v_node_us, v_node_vs, v_edges], 1) # (B, 3, D*num_heads)

        h, _ = self.multihead_attn(q, k, v)
        h_edge = h[:, -1, :]
        h_edge = self.edge_linear(h_edge)

        return h_edge

class TGNN(torch.nn.Module):
    def __init__(self, max_degree, num_node_classes, num_edge_classes, dim, num_steps, num_heads=[4, 4, 4, 1], dropout=0., norm='None', degree=False, augmented_features={}, **kwargs) -> None:
        super().__init__()
        self.max_degree = max_degree
        self.num_classes = num_edge_classes
        self.num_heads = num_heads 
        self.dim = dim
        self.num_steps = num_steps
        self.embedding_t = torch.nn.Linear(1, dim)
        self.time_pos_emb = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.layers = torch.nn.ModuleDict()
        self.norm = norm
        self.gru = torch.nn.Identity()
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.context_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )  

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.dropout = torch.nn.Dropout(p=dropout)
        if 'gru' in kwargs.keys():
            if kwargs['gru']:
                self.gru = torch.nn.GRU(dim, dim)

        for i, num_head in enumerate(num_heads):
            self.layers[f'time{i}'] = TimeEmb(dim, dim, Mish())
            self.layers[f'conv{i}'] = pyg.nn.TransformerConv(in_channels=dim*2, out_channels=dim, heads=num_head, concat=False)
            self.layers[f'norm{i}'] = norm_dict[self.norm](dim)
            self.layers[f'act{i}'] = torch.nn.SiLU()

        self.dummy_edge_feats = torch.nn.parameter.Parameter(torch.randn(dim))
        self.node_interaction = MiniAttentionLayer(node_dim=dim, in_edge_dim=dim, out_edge_dim=dim, d_model=dim, num_heads=2)
        
        self.final_out = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, self.num_classes)
        )
        
    def forward(self, pyg_data, t_node, t_edge,embedding=None):
        edge_attr_t = pyg_data.log_full_edge_attr_t.argmax(-1)
        is_edge_indices = edge_attr_t.nonzero(as_tuple=True)[0]
        
        edge_index = pyg_data.full_edge_index[:, is_edge_indices]
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
        nodes = pyg.utils.degree(edge_index[0],num_nodes=pyg_data.num_nodes).clamp(max=self.max_degree+1).long()

        nodes = nodes[..., None] / self.max_degree  # I prefer to make it embedding later
        # print("nodes:",nodes.shape)
        nodes = self.embedding_t(nodes)
        # print("embedding_t:",nodes.shape)
        t = self.time_pos_emb(t_node)
        t = self.mlp(t)
        
        h = nodes.unsqueeze(0)
        contexts = torch_scatter.scatter(nodes, pyg_data.batch, reduce='mean', dim=0)
        # print("contexts:",contexts.shape)
        contexts = self.global_mlp(contexts)

        contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)

        for i in range(len(self.num_heads)):
            ### add time embedding ###
            t_emb = self.layers[f'time{i}'](t)

            nodes = torch.cat([nodes, t_emb], dim=-1)
            
            ### message passing on graph ###
            nodes = self.layers[f'conv{i}'](nodes, edge_index)
            nodes = self.layers[f'norm{i}'](nodes)
            nodes = self.layers[f'act{i}'](nodes)
            nodes = self.dropout(nodes)

            ### gru update ### add some gate operation to merge with the origin nothing to do with sequence
            nodes, h = self.gru(nodes.unsqueeze(0).contiguous(), h.contiguous())
            h = self.dropout(h)
            nodes = nodes.squeeze(0)
            
            ### global context aggregation ###
            # aggregate locals to global
            node_contexts = self.context_mlp(torch.cat([nodes, contexts], dim=-1))
            contexts = torch_scatter.scatter(contexts + node_contexts, pyg_data.batch, reduce='mean', dim=0)
            contexts = self.global_mlp(contexts)
            contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)
            # spread global to locals
            nodes = nodes + contexts


        row, col = pyg_data.full_edge_index[0],  pyg_data.full_edge_index[1]
        edge_emb = torch.cat([nodes[row], nodes[col]], -1)
        edge_class = self.final_out(edge_emb)

        return pyg_data.log_node_attr, edge_class

class TGNN_degree_guided(torch.nn.Module):
    def __init__(self, max_degree, num_node_classes, num_edge_classes, dim, num_steps, num_heads=[4, 4, 4, 1], dropout=0., norm='None', degree=False, augmented_features={}, **kwargs) -> None:
        super().__init__()
        self.max_degree = max_degree
        self.num_classes = num_edge_classes
        self.num_heads = num_heads 
        self.dim = dim
        self.num_steps = num_steps
        self.embedding_t = torch.nn.Linear(1, dim)
        self.embedding_0 = torch.nn.Linear(1, dim)
        self.embedding_sel = torch.nn.Embedding(2, dim)
        self.node_in = torch.torch.nn.Sequential(
            torch.nn.Linear(dim * 3, dim),
            torch.nn.SiLU()
        )
        self.time_pos_emb = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.layers = torch.nn.ModuleDict()
        self.norm = norm
        self.gru = torch.nn.Identity()
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.context_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )  

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.dropout = torch.nn.Dropout(p=dropout)
        if 'gru' in kwargs.keys():
            if kwargs['gru']:
                self.gru = torch.nn.GRU(dim, dim)

        for i, num_head in enumerate(num_heads):
            self.layers[f'time{i}'] = TimeEmb(dim, dim, Mish())
            self.layers[f'conv{i}'] = pyg.nn.TransformerConv(in_channels=dim*2, out_channels=dim, heads=num_head, concat=False)
            self.layers[f'norm{i}'] = norm_dict[self.norm](dim)
            self.layers[f'act{i}'] = torch.nn.SiLU()

        self.dummy_edge_feats = torch.nn.parameter.Parameter(torch.randn(dim))

        self.node_out_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*4, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim*2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim*2, dim*2)
        )
        
        self.final_out = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, self.num_classes)
        )
        
    def forward(self, pyg_data, t_node, t_edge):
        if hasattr(pyg_data, 'edge_index_t'):
            edge_index = pyg_data.edge_index_t
        else: 
            edge_attr_t = pyg_data.log_full_edge_attr_t.argmax(-1)
            is_edge_indices = edge_attr_t.nonzero(as_tuple=True)[0]

            edge_index = pyg_data.full_edge_index[:, is_edge_indices]
            edge_index = torch.cat([edge_index, edge_index.flip(0)],dim=-1)

        nodes_t = pyg.utils.degree(edge_index[0],num_nodes=pyg_data.num_nodes).clamp(max=self.max_degree+1).long()
        # print("nodes_t:",nodes_t.shape)
        node_selection = torch.zeros_like(nodes_t)


        nodes_t = nodes_t[..., None] / self.max_degree  # I prefer to make it embedding later
        nodes_0 = pyg_data.degree[..., None] / self.max_degree
        node_selection[pyg_data.active_node_indices] = 1
        node_selection = node_selection.long()
        nodes = torch.cat([self.embedding_t(nodes_t), self.embedding_0(nodes_0), self.embedding_sel(node_selection)], dim=-1)
        # print("nodes:",nodes.shape)
        nodes = self.node_in(nodes)

        t = self.time_pos_emb(t_node)
        t = self.mlp(t)
        
        h = nodes.unsqueeze(0)
        # print("nodes-context:",nodes.shape)
        # print("nodes-context-batch:",pyg_data.batch)
        contexts = torch_scatter.scatter(nodes, pyg_data.batch, reduce='mean', dim=0)
        # print("contexts:",contexts.shape)s
        contexts = self.global_mlp(contexts)

        contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)

        for i in range(len(self.num_heads)):
            ### add time embedding ###
            t_emb = self.layers[f'time{i}'](t)

            nodes = torch.cat([nodes, t_emb], dim=-1)
            
            ### message passing on graph ###
            nodes = self.layers[f'conv{i}'](nodes, edge_index)
            nodes = self.layers[f'norm{i}'](nodes)
            nodes = self.layers[f'act{i}'](nodes)
            nodes = self.dropout(nodes)

            ### gru update ###
            nodes, h = self.gru(nodes.unsqueeze(0).contiguous(), h.contiguous())
            h = self.dropout(h)
            nodes = nodes.squeeze(0)
            
            ### global context aggregation ###
            # aggregate locals to global
            node_contexts = self.context_mlp(torch.cat([nodes, contexts], dim=-1))
            contexts = torch_scatter.scatter(contexts + node_contexts, pyg_data.batch, reduce='mean', dim=0)
            contexts = self.global_mlp(contexts)
            contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)
            # spread global to locals
            nodes = nodes + contexts

        # mlp add
        row = pyg_data.full_edge_index[0].index_select(0, pyg_data.active_edge_indices)
        col = pyg_data.full_edge_index[1].index_select(0, pyg_data.active_edge_indices)

        nodes = torch.cat([nodes, self.embedding_t(nodes_t), self.embedding_0(nodes_0), self.embedding_sel(node_selection)], dim=-1)
        nodes = self.node_out_mlp(nodes)

        edge_emb = nodes[row] + nodes[col]
        edge_class = self.final_out(edge_emb)
        
        return pyg_data.log_node_attr, edge_class


class TGNN_embedding_guided(torch.nn.Module):
    def __init__(self, max_degree, num_node_classes, num_edge_classes, dim, num_steps,embedding_dim=256,num_heads=[4, 4, 4, 1], dropout=0., norm='None', degree=False, augmented_features={}, **kwargs) -> None:
        super().__init__()
        self.max_degree = max_degree
        self.num_classes = num_edge_classes
        self.num_heads = num_heads
        self.dim = dim
        self.num_steps = num_steps
        self.embedding_t = torch.nn.Linear(1, dim)
        self.embedding_0 = torch.nn.Linear(1, dim)
        self.node_in = torch.torch.nn.Sequential(
            torch.nn.Linear(dim * 2, dim),
            torch.nn.SiLU()
        )
        self.time_pos_emb = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.layers = torch.nn.ModuleDict()
        self.norm = norm
        self.gru = torch.nn.Identity()
        self.global_mlp=torch.nn.Linear(1, dim)

        self.context_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim*4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )  

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim)
            )

        self.dropout = torch.nn.Dropout(p=dropout)
        if 'gru' in kwargs.keys():
            if kwargs['gru']:
                self.gru = torch.nn.GRU(dim, dim)

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2),
            torch.nn.SiLU(),
        )

        for i, num_head in enumerate(num_heads):
            self.layers[f'time{i}'] = TimeEmb(dim, dim, Mish())
            self.layers[f'conv{i}'] = pyg.nn.TransformerConv(in_channels=dim*2, out_channels=dim, heads=num_head, concat=False)
            self.layers[f'norm{i}'] = norm_dict[self.norm](dim)
            self.layers[f'act{i}'] = torch.nn.SiLU()

        self.dummy_edge_feats = torch.nn.parameter.Parameter(torch.randn(dim))

        self.node_out_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*3, dim * 2),
            torch.nn.SiLU(),
        )
        
        self.final_out = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, self.num_classes)
        )
        self.gae= torch.load("/home/hsy/AGGS/gae/weight/elliptic_gae_relu_256.pt")
        # 自适应池化
        self.adapter = torch.nn.AdaptiveAvgPool1d(embedding_dim)
        # 将context——node降维
        self.context_node=torch.nn.Linear(dim,1)

        
    def forward(self, pyg_data, t_node, t_edge,embedding=None):
        #pyg_data: DataBatch(edge_index=[2, 324], feature=[418, 166], label=[418], num_nodes=418, full_edge_index=[2, 254], full_edge_attr=[254], node_attr=[418], degree=[251], feature0=[418], feature1=[418], feature2=[418], feature3=[418], feature4=[418], feature5=[418], feature6=[418], feature7=[418], feature8=[418], feature9=[418], feature10=[418], feature11=[418], feature12=[418], feature13=[418], feature14=[418], feature15=[418], feature16=[418], feature17=[418], feature18=[418], feature19=[418], feature20=[418], feature21=[418], feature22=[418], feature23=[418], feature24=[418], feature25=[418], feature26=[418], feature27=[418], feature28=[418], feature29=[418], feature30=[418], feature31=[418], feature32=[418], feature33=[418], feature34=[418], feature35=[418], feature36=[418], feature37=[418], feature38=[418], feature39=[418], feature40=[418], feature41=[418], feature42=[418], feature43=[418], feature44=[418], feature45=[418], feature46=[418], feature47=[418], feature48=[418], feature49=[418], feature50=[418], feature51=[418], feature52=[418], feature53=[418], feature54=[418], feature55=[418], feature56=[418], feature57=[418], feature58=[418], feature59=[418], feature60=[418], feature61=[418], feature62=[418], feature63=[418], feature64=[418], feature65=[418], feature66=[418], feature67=[418], feature68=[418], feature69=[418], feature70=[418], feature71=[418], feature72=[418], feature73=[418], feature74=[418], feature75=[418], feature76=[418], feature77=[418], feature78=[418], feature79=[418], feature80=[418], feature81=[418], feature82=[418], feature83=[418], feature84=[418], feature85=[418], feature86=[418], feature87=[418], feature88=[418], feature89=[418], feature90=[418], feature91=[418], feature92=[418], feature93=[418], feature94=[418], feature95=[418], feature96=[418], feature97=[418], feature98=[418], feature99=[418], feature100=[418], feature101=[418], feature102=[418], feature103=[418], feature104=[418], feature105=[418], feature106=[418], feature107=[418], feature108=[418], feature109=[418], feature110=[418], feature111=[418], feature112=[418], feature113=[418], feature114=[418], feature115=[418], feature116=[418], feature117=[418], feature118=[418], feature119=[418], feature120=[418], feature121=[418], feature122=[418], feature123=[418], feature124=[418], feature125=[418], feature126=[418], feature127=[418], feature128=[418], feature129=[418], feature130=[418], feature131=[418], feature132=[418], feature133=[418], feature134=[418], feature135=[418], feature136=[418], feature137=[418], feature138=[418], feature139=[418], feature140=[418], feature141=[418], feature142=[418], feature143=[418], feature144=[418], feature145=[418], feature146=[418], feature147=[418], feature148=[418], feature149=[418], feature150=[418], feature151=[418], feature152=[418], feature153=[418], feature154=[418], feature155=[418], feature156=[418], feature157=[418], feature158=[418], feature159=[418], feature160=[418], feature161=[418], feature162=[418], feature163=[418], feature164=[418], feature165=[418], batch=[418], ptr=[257], nodes_per_graph=[256], edges_per_graph=[256], log_node_attr=[418, 2], log_full_edge_attr=[254, 2], log_node_attr_tmin1=[418, 2], log_full_edge_attr_tmin1=[254, 2], log_node_attr_t=[418, 2], log_full_edge_attr_t=[254, 2], active_node_indices=[2], active_edge_indices=[1], edge_predict_masks=[254])
        if hasattr(pyg_data, 'edge_index_t'):
            edge_index = pyg_data.edge_index_t
        else: 
            edge_attr_t = pyg_data.log_full_edge_attr_t.argmax(-1)
            is_edge_indices = edge_attr_t.nonzero(as_tuple=True)[0]

            edge_index = pyg_data.full_edge_index[:, is_edge_indices]
            edge_index = torch.cat([edge_index, edge_index.flip(0)],dim=-1)
        
        # 计算t时刻的子图embedding:[12,256]
        # nodes_t=self.gae.encode(pyg_data.feature,edge_index)
        nodes_t = self.gae.encode(
            pyg_data.feature.to(dtype=torch.float),  # 转换数据类型
            edge_index
        )

        # 计算每一维度nodes的max值作为子图embedding：[256]
        subgraph_t= torch.max(nodes_t, dim=0).values
        # 展平：subgraph_t: torch.Size([256, 1])
        subgraph_t = subgraph_t.unsqueeze(1)

        # 获取0时刻的子图embedding[12,256]
        if embedding is not None:
            subgraph_0=embedding
        else:
            node_0=pyg_data.embedding
            subgraph_0= torch.max(node_0, dim=0).values
        subgraph_0 = subgraph_0.unsqueeze(1)
        # 处理0和t时刻的子图embedding
        # embedding_t:[256,64]
        embedding_t=self.embedding_t(subgraph_t)
        # embedding_0:[256,64]
        embedding_0=self.embedding_0(subgraph_0)
        # 拼接0和t时刻的子图特征:[256,128]
        nodes = torch.cat([embedding_0,embedding_t], dim=-1)
        # final-nodes: torch.Size([256, 64])
        nodes = self.node_in(nodes)

        # 时间编码：t: torch.Size([13, 64])
        t = self.time_pos_emb(t_node)
        t = self.mlp(t)

        # 子图全局特征：子图embedding的mean来进行计算
        contexts=torch.mean(nodes, dim=1)
        contexts=contexts.unsqueeze(1)
        contexts = self.global_mlp(contexts)
        
        # 展平:h: torch.Size([1, 256, 64])
        h = nodes.unsqueeze(0)

        for i in range(len(self.num_heads)):
            ### add time embedding ###
            # [14,64]
            t_emb = self.layers[f'time{i}'](t)
            nodes = torch.cat([nodes, t_emb], dim=0)
            nodes=self.node_mlp(nodes)
            
            ### message passing on graph ###
            nodes = self.layers[f'conv{i}'](nodes, edge_index)
            # print("conv:",nodes.shape)
            nodes = self.layers[f'norm{i}'](nodes)
            # print("norm:",nodes.shape)
            nodes = self.layers[f'act{i}'](nodes)
            nodes = self.dropout(nodes)
            # print("final:",nodes.shape)

            ### gru update ###
            nodes = nodes.unsqueeze(0)
            nodes = torch.transpose(nodes, 1, 2)
            nodes=self.adapter(nodes)
            nodes = torch.transpose(nodes, 1, 2)
            nodes, h = self.gru(nodes.contiguous(), h.contiguous())
            h = self.dropout(h)
            nodes = nodes.squeeze(0)
            
            ### global context aggregation ###
            node_contexts = self.context_mlp(torch.cat([nodes, contexts], dim=-1))
            contexts=self.context_node(node_contexts)
            contexts = self.global_mlp(contexts)
            # spread global to locals
            nodes = nodes + contexts

        # mlp add
        row = pyg_data.full_edge_index[0]
        col = pyg_data.full_edge_index[1]
        
        nodes = torch.cat([nodes, embedding_t,embedding_0], dim=-1)
        # embedding_0:[256,64]
        nodes = self.node_out_mlp(nodes)
        # 复制nodes
        max_row = torch.max(row)
        max_col = torch.max(col)
        # 返回两个最大值中的较大者
        max_value = torch.max(max_row, max_col)
        repeat_num=max_value.item()//nodes.size(0)+1
        # repeat
        nodes_repeated = torch.cat([nodes for i in range(repeat_num)], dim=0)
        edge_emb = nodes_repeated[row] + nodes_repeated[col]
        edge_class = self.final_out(edge_emb)
        return pyg_data.log_node_attr, edge_class