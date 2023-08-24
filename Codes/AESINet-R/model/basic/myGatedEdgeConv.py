import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d
from torch_geometric.nn.conv import MessagePassing
from torch_cluster import knn_graph
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops

def map(data,MIN,MAX):
    d_min = torch.max(data)    
    d_max = torch.min(data)    
    return MIN + (MAX-MIN)/(d_max-d_min + 0.0001) * (data - d_min)


class Gatedgcn(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Gatedgcn, self).__init__(aggr='add')  # "Max" aggregation. ['add', 'mean', 'max']
        self.linA = torch.nn.Linear(in_channels, out_channels)
        self.linB = torch.nn.Linear(in_channels, out_channels)
        self.linU = torch.nn.Linear(in_channels, out_channels)
        self.linV = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x0 has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):  # massage flow from node_j to node_i
        # # x_i has shape [E, in_channels]
        # # x_j has shape [E, in_channels]
        etaij = torch.sigmoid(self.linA(x_i) + self.linB(x_j))
        # self.linU(x_i) + etaij * self.linV(x_j)
        return etaij * self.linV(x_j)

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        return aggr_out + self.linU(x)


############################################################################
class GatedEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GatedEdgeConv, self).__init__(aggr='add')  # "Max" aggregation. ['add', 'mean', 'max']
        self.linB = torch.nn.Linear(in_channels, out_channels)
        self.linA = torch.nn.Linear(in_channels, out_channels)
        self.mlp = Seq(Linear(out_channels, 1 * out_channels),
                       ReLU(),
                       Linear(1 * out_channels, out_channels))
        self.sigma = 0.1 * 3.1415926  # 3.1415926*0.1

    def forward(self, x0, edge_index, edge_attr_dist):
        # x0 has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_index, _ = remove_self_loops(edge_index)
        x = self.linB(x0)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, x0=x0, edge_attr_dist=edge_attr_dist)

    def message(self, x_i, x_j, edge_attr_dist):  # massage flow from node_j to node_i
        # # x_i has shape [E, in_channels]
        # # x_j has shape [E, in_channels]
        # # edge_attr_dist has shape E, we reshape it as [E, 1]
        edge_attr_dist = edge_attr_dist.view(-1, 1)
        # dist = torch.log(edge_attr_dist)*(-self.sigma ** 2)
        # distNorm = dist / torch.max(dist)
        # tmp = torch.cat([(x_j-x_i) * distNorm * 10], dim=1)  # tmp has shape [E, in_channels]
        # tmp_g = (self.mlp(tmp).abs() + 0.000001).pow(-1.0)
        # gate = (2 * torch.sigmoid(tmp_g) - 1)
        tmp = torch.cat([(x_j-x_i) * edge_attr_dist], dim=1)  # tmp has shape [E, in_channels]
        # tmp = torch.cat([(x_j-x_i)], dim=1)  # tmp has shape [E, in_channels]
        gate = self.mlp(tmp)
        return gate * x_j

    def update(self, aggr_out, x0):
        # aggr_out has shape [N, out_channels]

        return aggr_out + self.linA(x0)

############################################################################
class AdGatedEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(AdGatedEdgeConv, self).__init__(aggr='add')  # "Max" aggregation. ['add', 'mean', 'max']
        self.linB = torch.nn.Linear(in_channels, out_channels)
        self.linA = torch.nn.Linear(in_channels, out_channels)
        # self.mlp = Seq(Linear(2*out_channels, 1 * out_channels),
        #                ReLU(),
        #                Linear(1 * out_channels, out_channels))
        self.mlp = Seq(Linear(2 * out_channels, 1 * out_channels))
        # self.sigma = 0.1 * 3.1415926  # 3.1415926*0.1

    def forward(self, x0, edge_index, edge_attr_dist):
        # x0 has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_index, _ = remove_self_loops(edge_index)
        x = self.linB(x0)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, x0=x0, edge_attr_dist=edge_attr_dist)

    def message(self, x_i, x_j, edge_attr_dist):  # massage flow from node_j to node_i
        # # x_i has shape [E, in_channels]
        # # x_j has shape [E, in_channels]
        # # edge_attr_dist has shape E, we reshape it as [E, 1]
        edge_attr_dist = edge_attr_dist.view(-1, 1)
        # dist = torch.log(edge_attr_dist)*(-self.sigma ** 2)
        # distNorm = dist / torch.max(dist)
        # tmp = torch.cat([(x_j-x_i) * distNorm * 10], dim=1)  # tmp has shape [E, in_channels]
        # tmp_g = (self.mlp(tmp).abs() + 0.000001).pow(-1.0)
        # gate = (2 * torch.sigmoid(tmp_g) - 1)
        # tmp = torch.cat([(x_j-x_i) * edge_attr_dist], dim=1)  # tmp has shape [E, in_channels]
        tmp = torch.cat((x_j, x_i), dim=1) * edge_attr_dist # tmp has shape [E, in_channels]
        gate = 2 * torch.sigmoid(self.mlp(tmp) - 1)
        return gate * x_j

    def update(self, aggr_out, x0):
        # aggr_out has shape [N, out_channels]
        return aggr_out + self.linA(x0)
######################################################################################


class DynamicGatedEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(DynamicGatedEdgeConv, self).__init__(aggr='add')  # "Max" aggregation. ['add', 'mean', 'max']
        self.k = 32
        self.linB = torch.nn.Linear(in_channels, out_channels)
        self.linA = torch.nn.Linear(in_channels, out_channels)
        self.mlp = Seq(Linear(out_channels, 2 * out_channels),
                       ReLU(),
                       Linear(2 * out_channels, out_channels))
        # self.sigma = 0.1 * 3.1415926  # 3.1415926*0.1

    def forward(self, x0, batch=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_index, _ = remove_self_loops(edge_index)
        x = self.linB(x0)
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, x0=x0)

    def message(self, x_i, x_j):  # massage flow from node_j to node_i
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        tmp = torch.cat([(x_j-x_i)], dim=1)  # tmp has shape [E, in_channels]
        gate = self.mlp(tmp)
        return gate * x_j

    def update(self, aggr_out, x0):
        # aggr_out has shape [N, out_channels]

        return aggr_out + self.linA(x0)
######################################################################################


class NodeAtt(nn.Module):
    def __init__(self, in_channels):
        super(NodeAtt, self).__init__()
        self.mlp = Seq(Linear(1 * in_channels, in_channels),
                       ReLU(),
                       Linear(in_channels, 1))
        self.lin = Linear(1 * in_channels, in_channels)

    def forward(self, x_S, x_D):
        # x_S has shape [N, in_channels]
        # x_D has shape [N, in_channels]
        # x = torch.cat([x_S, x_D], dim=1)  # x has shape [N, 2*in_channels]
        x = x_S + x_D  # x has shape [N, 1*in_channels]
        nodeatt = torch.sigmoid(self.mlp(x))  # has shape [N, 1]
        x_out = self.lin(x * nodeatt)  # [N, in_channels]
        return x_out


class NodeAtt_wo(nn.Module):
    def __init__(self, in_channels):
        super(NodeAtt_wo, self).__init__()

    def forward(self, x_S, x_D):
        # x_S has shape [N, in_channels]
        # x_D has shape [N, in_channels]
        # x = torch.cat([x_S, x_D], dim=1)  # x has shape [N, 2*in_channels]
        x_out = x_S + x_D  # x has shape [N, 1*in_channels]
        return x_out