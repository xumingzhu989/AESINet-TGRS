import torch
from torch import nn
import matplotlib.pyplot as plt
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F


# 训练过程中显示显著图和真值图
def img_show(M):
    a = M[0].cpu()
    aa = a.detach().numpy()
    b = M[1].cpu()
    bb = b.detach().numpy()
    plt.subplot(121)
    plt.imshow(aa)
    plt.subplot(122)
    plt.imshow(bb)
    plt.show()


# 将CNN的批处理数据转换为PYG中的批处理数据格式，以一个图像块作为基本处理单元,也即只有一个图像块内的节点才去交互
def Generate_edges_inside_a_block(adj, bndnum, thr, B, x_t, node_num, bknum):  # x is [B, C, node_num]
    edge_index = []
    edge_attr = []
    data = []
    batch = []
    adj = torch.where(adj < thr, torch.zeros_like(adj), adj)  # 将adj中相似度值较小的置为0
    adj_new = torch.zeros_like(adj)
    for k in range(bknum*bknum):
        adj_new[:, k*bndnum:(k+1)*bndnum, k*bndnum:(k+1)*bndnum] = adj[:, k*bndnum:(k+1)*bndnum, k*bndnum:(k+1)*bndnum]
    # img_show(adj_new)
    for i in range(B):
        edge_index_i, edge_attr_i = dense_to_sparse(adj_new[i])  # [2, node_connect], [node_connect]
        edge_index.append(edge_index_i + i*node_num)
        edge_attr.append(edge_attr_i)
        data.append(x_t[i])  # x_t[i] is [node_num, C]
        batch.append(torch.LongTensor(node_num).fill_(i))
    graph_data = torch.cat(data, dim=0)
    graph_edge_index = torch.cat(edge_index, dim=1)
    graph_edge_attr = torch.cat(edge_attr, dim=0)
    graph_batch = torch.cat(batch, dim=0).to(adj.device)
    return graph_data, graph_edge_index, graph_edge_attr, B, graph_batch  
    # [node_num_batch, C], [2, node_connect_batch], [node_connect_batch]


# 将CNN的批处理数据转换为PYG中的批处理数据格式，以四个块为一个单元进行处理
def Generate_edges_between_blocks(adj, bndnum, thr, B, x_t, node_num, bknum):  # x is [B, C, node_num]
    edge_index = []
    edge_attr = []
    data = []
    batch = []
    adj = torch.where(adj < thr, torch.zeros_like(adj), adj)  # 将adj中相似度值较小的置为0
    adj_new = torch.zeros_like(adj)
    for k in range(int(bknum * bknum / 2)):
        k = k*2
        adj_new[:, k * bndnum:(k + 2) * bndnum, k * bndnum:(k + 2) * bndnum] = adj[:, k * bndnum:(k + 2) * bndnum, k * bndnum:(k + 2) * bndnum]
    adj_new[:, 0 * bndnum:(0 + 2) * bndnum, 4 * bndnum:(4 + 2) * bndnum] = adj[:, 0 * bndnum:(0 + 2) * bndnum, 4 * bndnum:(4 + 2) * bndnum]
    adj_new[:, 2 * bndnum:(2 + 2) * bndnum, 6 * bndnum:(6 + 2) * bndnum] = adj[:, 2 * bndnum:(2 + 2) * bndnum, 6 * bndnum:(6 + 2) * bndnum]
    adj_new[:, 4 * bndnum:(4 + 2) * bndnum, 0 * bndnum:(0 + 2) * bndnum] = adj[:, 4 * bndnum:(4 + 2) * bndnum, 0 * bndnum:(0 + 2) * bndnum]
    adj_new[:, 6 * bndnum:(6 + 2) * bndnum, 2 * bndnum:(2 + 2) * bndnum] = adj[:, 6 * bndnum:(6 + 2) * bndnum, 2 * bndnum:(2 + 2) * bndnum]
    adj_new[:, 8 * bndnum:(8 + 2) * bndnum, 12 * bndnum:(12 + 2) * bndnum] = adj[:, 8 * bndnum:(8 + 2) * bndnum, 12 * bndnum:(12 + 2) * bndnum]
    adj_new[:, 10 * bndnum:(10 + 2) * bndnum, 14 * bndnum:(14 + 2) * bndnum] = adj[:, 10 * bndnum:(10 + 2) * bndnum, 14 * bndnum:(14 + 2) * bndnum]
    adj_new[:, 12 * bndnum:(12 + 2) * bndnum, 8 * bndnum:(8 + 2) * bndnum] = adj[:, 12 * bndnum:(12 + 2) * bndnum, 8 * bndnum:(8 + 2) * bndnum]
    adj_new[:, 14 * bndnum:(14 + 2) * bndnum, 10 * bndnum:(10 + 2) * bndnum] = adj[:, 14 * bndnum:(14 + 2) * bndnum, 10 * bndnum:(10 + 2) * bndnum]
    # img_show(adj_new)
    for i in range(B):
        edge_index_i, edge_attr_i = dense_to_sparse(adj_new[i])  # [2, node_connect], [node_connect]
        edge_index.append(edge_index_i + i * node_num)
        edge_attr.append(edge_attr_i)
        data.append(x_t[i])  # x_t[i] is [node_num, C]
        batch.append(torch.LongTensor(node_num).fill_(i))
    graph_data = torch.cat(data, dim=0)
    graph_edge_index = torch.cat(edge_index, dim=1)
    graph_edge_attr = torch.cat(edge_attr, dim=0)
    graph_batch = torch.cat(batch, dim=0).to(adj.device)
    return graph_data, graph_edge_index, graph_edge_attr, B, graph_batch  # [node_num_batch, C], [2, node_connect_batch], [node_connect_batch]

# 选取邻接矩阵中第k大的值作为阈值，生成k(近似为k，因为在有些矩阵值相等的时候会有所出入)个最近的邻居节点
def thr_select(adj, k):  # adj是B*N*N的邻接矩阵，k是取邻居节点个数
    B, N, N = adj.size()
    adj_flat = adj.view(B,-1)  # B*(N*N)
    adj_uniq = torch.unique(adj_flat,dim=1)
    val, _ = adj_uniq.topk(k, dim=1, largest=True, sorted=True)  # val is B*k
    thr = val[:,k-1]  # 取第k个值作为阈值，维度为B
    return thr.unsqueeze(1).unsqueeze(2)  # 维度为B*1*1


# 将CNN的批处理数据转换为PYG中的批处理数据格式，以全局作为处理单元
def Generate_edges_globally(adj, bndnum, thr, B, x_t, node_num, bknum):  # x is [B, C, node_num]
    edge_index = []
    edge_attr = []
    data = []
    batch = []
    adj = torch.where(adj < thr, torch.zeros_like(adj), adj)  # 将adj中相似度值较小的置为0
    for i in range(B):
        edge_index_i, edge_attr_i = dense_to_sparse(adj[i])  # [2, node_connect], [node_connect]
        edge_index.append(edge_index_i + i*node_num)
        edge_attr.append(edge_attr_i)
        data.append(x_t[i])  # x_t[i] is [node_num, C]
        batch.append(torch.LongTensor(node_num).fill_(i))
    graph_data = torch.cat(data, dim=0)
    graph_edge_index = torch.cat(edge_index, dim=1)
    graph_edge_attr = torch.cat(edge_attr, dim=0)
    graph_batch = torch.cat(batch, dim=0).to(adj.device)
    return graph_data, graph_edge_index, graph_edge_attr, B, graph_batch  # [node_num_batch, C], [2, node_connect_batch], [node_connect_batch]


# 将PYG的批处理数据再转换为CNN中的批处理数据格式
def graph_batch_re_trans(x, B, node_num_batch, C_new):  # x is [node_num_batch, C_new]
    node_num = int(node_num_batch/B)
    data = torch.zeros([B, node_num, C_new], device=x.device, dtype=x.dtype, layout=x.layout)
    for i in range(B):
        ind_s = i*node_num
        ind_e = (i+1)*node_num
        data[i] = x[ind_s:ind_e,:]
    data_o = data.permute(0, 2, 1).contiguous()  # 由[B, node_num_batch, C_new]→[B, C_new, node_num_batch]
    return data_o  # [B, C_new, node_num_batch]


