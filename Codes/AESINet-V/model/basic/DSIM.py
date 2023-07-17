import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, SGConv, GATConv, EGConv, BatchNorm, global_max_pool, global_mean_pool
from model.basic.graphbatch import Generate_edges_globally, Generate_edges_between_blocks, Generate_edges_inside_a_block, graph_batch_re_trans
from model.basic.myGatedEdgeConv import Gatedgcn, GatedEdgeConv, AdGatedEdgeConv
from model.basic.Attention import *
import math


class GraphInference(nn.Module):
    def __init__(self, dim, loop, bknum, thr4, thr2, thr1):
        super(GraphInference, self).__init__()
        self.thr4 = thr4
        self.thr2 = thr2
        self.thr1 = thr1
        self.bknum = bknum
        self.loop = loop
        self.gcn1 = AdGatedEdgeConv(dim, dim)
        self.gcn2 = AdGatedEdgeConv(dim, dim)
        self.gcn3 = AdGatedEdgeConv(dim, dim)
        self.gcns = [self.gcn1, self.gcn2, self.gcn3]

        self.bn1 = BatchNorm(dim)
        self.bn2 = BatchNorm(dim)
        self.bn3 = BatchNorm(dim)
        self.bns = [self.bn1, self.bn2, self.bn3]

        self.Att1 = NodeAtt(dim)
        self.Att2 = NodeAtt(dim)
        self.Att3 = NodeAtt(dim)
        self.Atts = [self.Att1, self.Att2, self.Att3]

        assert (loop == 0 or loop == 1 or loop == 2 or loop == 3)
        self.gcns = self.gcns[0:loop]
        self.bns = self.bns[0:loop]
        self.Atts = self.Atts[0:loop]

        self.relu = nn.ReLU()
        self.lineA4 = torch.nn.Linear(dim, 1)
        self.lineA2 = torch.nn.Linear(dim, 1)
        self.lineA1 = torch.nn.Linear(dim, 1)
        self.mlpCA4 = torch.nn.Sequential(torch.nn.Linear(dim, dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(dim, dim))
        self.mlpCA2 = torch.nn.Sequential(torch.nn.Linear(dim, dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(dim, dim))
        self.mlpCA1 = torch.nn.Sequential(torch.nn.Linear(dim, dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(dim, dim))
        self.lineA = torch.nn.Linear(3, 3)
        self.lineFu = torch.nn.Linear(dim, dim)

    def forward(self, x):  # x is [B, C, node_num]
        B, C, node_num = x.size()
        bndnum = int(node_num / (self.bknum * self.bknum))
        x = F.normalize(x, p=2, dim=1)  # 即对每个节点按照通道维度C进行2范数Norm，每个节点的所有通道值均在0-1之间；
        x_t = x.permute(0, 2, 1).contiguous()  # x_t is [B, node_num, C]
        # adj = euc_dist(x_t)  # dist_adj is [B, node_num, node_num]
        adj = (torch.matmul(x_t, x))  # adj is [B, node_num, node_num]
        # 图卷积运算
        graph_data4, edge_index4, edge_attr4, B, graph_batch4 = Generate_edges_inside_a_block(adj, bndnum, self.thr4, B, x_t, node_num, self.bknum)  
        # x is [B, C, node_num], output is [node_num_batch, C], [2, node_connect_batch], [node_connect_batch]
        y4 = self.relu(self.Atts[0](self.bns[0](self.gcns[0](graph_data4, edge_index4, edge_attr4)) + graph_data4))

        graph_data2, edge_index2, edge_attr2, B, graph_batch2 = Generate_edges_between_blocks(adj, bndnum, self.thr2, B, x_t, node_num, self.bknum)  
        # x is [B, C, node_num], output is [node_num_batch, C], [2, node_connect_batch], [node_connect_batch]
        y2 = self.relu(self.Atts[1](self.bns[1](self.gcns[1](graph_data2, edge_index2, edge_attr2)) + graph_data2))

        graph_data1, edge_index1, edge_attr1, B, graph_batch1 = Generate_edges_globally(adj, bndnum, self.thr1, B, x_t, node_num, self.bknum)  
        # x is [B, C, node_num], output is [node_num_batch, C], [2, node_connect_batch], [node_connect_batch]
        y1 = self.relu(self.Atts[2](self.bns[2](self.gcns[2](graph_data1, edge_index1, edge_attr1)) + graph_data1))

        w4 = torch.sigmoid(self.mlpCA4(global_max_pool(y4, graph_batch4)))  # B*C_new，channel-wise Attention     
        w2 = torch.sigmoid(self.mlpCA2(global_max_pool(y2, graph_batch2)))
        w1 = torch.sigmoid(self.mlpCA1(global_max_pool(y1, graph_batch1)))

        a4 = self.lineA4(w4)  # B*1，用来对不同的图卷积进行加权
        a2 = self.lineA2(w2)
        a1 = self.lineA1(w1)
        A = torch.cat((a4, a2, a1), dim=1)  # 不同图卷积权值和为1
        A = F.softmax(A, dim=1)
        W4 = w4[graph_batch4]  # (B*node_num)*C_new
        A4 = A[:, 0][graph_batch4]  # (B*node_num)*1
        W2 = w2[graph_batch2]
        A2 = A[:, 1][graph_batch2]  # (B*node_num)*1
        W1 = w1[graph_batch1]
        A1 = A[:, 2][graph_batch1]  # (B*node_num)*1
        y = self.lineFu(A4.unsqueeze(1) * W4 * y4 + A2.unsqueeze(1) * W2 * y2 + A1.unsqueeze(1) * W1 * y1)
        node_num_batch, C_new = y.size()
        output = graph_batch_re_trans(y, B, node_num_batch, C_new)  # trans from [node_num_batch, C_new] to [B, C_new, node_num]
        return output  # [B, C_new, node_num]







class GraphProjection(nn.Module):
    def __init__(self,bnum,bnod,dim,normalize_input=False):
        super(GraphProjection, self).__init__()
        self.bnum=bnum      #横向或纵向分割块数，则图像块总数量：bnum x bnum
        self.bnod=bnod      #单个图像块映射图节点数量
        self.node_num=bnum*bnum*bnod
        self.dim = dim
        self.normalize_input = normalize_input
        self.anchor = nn.Parameter(torch.rand(self.node_num, dim))
        self.sigma = nn.Parameter(torch.rand(self.node_num, dim))
        #随机初始化w与σ

    def gen_soft_assign(self, x, sigma):#计算软分配矩阵
        B, C, H, W = x.size() # x的维度C需要与self.dim大小一致
        soft_assign = torch.zeros([B, self.node_num, self.n], device=x.device, dtype=x.dtype, layout=x.layout)#这一步是初始化软分配矩阵,torch.zeros() 返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor
        soft_ass = torch.zeros([B, self.node_num, self.n], device=x.device, dtype=x.dtype, layout=x.layout)
        for node_id in range(self.node_num):
            block_id=math.floor(node_id/self.bnod)
            h_sta=math.floor(block_id/self.bnum)*self.h
            w_sta=block_id%(self.bnum)*self.w
            h_end=h_sta+self.h
            w_end=w_sta+self.w
            tmp=x.view(B, C, H, W)[: , : ,h_sta:h_end , w_sta : w_end]
            tmp=tmp.reshape(B,C,-1).permute(0,2,1).contiguous()
            residual = (tmp - self.anchor[node_id, :]).div(sigma[node_id, :]) # + eps)
            soft_assign[:, node_id, :] = -torch.pow(torch.norm(residual, dim=2),2)/2     #torch.norm(tensor,dim)按dim维度求二范数
            # soft_assign[:, node_id, :] = -torch.norm(residual, dim=2)/2     #torch.norm(tensor,dim)按dim维度求二范数 
        for block_id in range(self.bnum*self.bnum):
            node_sta=self.bnod*block_id
            node_end=node_sta+self.bnod
            soft_ass[: , node_sta:node_end , :]=F.softmax(soft_assign[: , node_sta:node_end , :], dim=1)#对矩阵计算软赋值，形成最终的软分配矩阵
        return soft_ass #B node_num n

    def forward(self, x):
        B, C, H, W = x.size()
        self.h=math.floor(H/self.bnum)
        self.w=math.floor(W/self.bnum)
        self.n=self.h*self.w
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)#按照1维度计算x的2范数 #across descriptor dim
        sigma = torch.sigmoid(self.sigma)   #用sigmoid函数对σ处理
        soft_assign = self.gen_soft_assign(x, sigma)# 计算soft-assignment矩阵 # B x node_num x N(N=HxW) 
        eps = 1e-9
        nodes = torch.zeros([B, self.node_num, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for node_id in range(self.node_num):
            block_id=math.floor(node_id/self.bnod)
            h_sta=math.floor(block_id/self.bnum)*self.h
            w_sta=block_id%(self.bnum)*self.w
            h_end=h_sta+self.h
            w_end=w_sta+self.w
            tmp=x.view(B, C, H, W)[: , : ,h_sta:h_end , w_sta : w_end]
            tmp=tmp.reshape(B,C,-1).permute(0,2,1).contiguous()
            residual = (tmp - self.anchor[node_id, :]).div(sigma[node_id, :]) # + eps)
            #转置之后的x：  B x H x W x c                                     B x s x s x C
            #计算(x_ij − w_k) /σ_k
            nodes[:, node_id, :] = residual.mul(soft_assign[:, node_id, :].unsqueeze(2)).sum(dim=1) / (soft_assign[:, node_id, :].sum(dim=1).unsqueeze(1) + eps)
            # nodes[:, node_id, :] = residual.mul(soft_assign[:, node_id, :].unsqueeze(2)).sum(dim=1)
            #        [B,C]         [B,n,C]              [B,n]              [B,n,1]       [B,C]                [B,n]                [B]        [B,1]
            #mul那一步是以[B,HW,C]与[B,HW,1]同位置相乘（与矩阵乘不同），以此来用概率值进行加权求平均                                            
            #unsqueeze(2)在第1维和第2维之间增加一个维度，使其变成新的第2维，此时原第2维变成第3维
            #tensor.sum(dim)对dim维度进行求和，此后dim维度所有的值变成了一个值，dim维度消失
            #多维矩阵乘：除了最后两维，其它维的宽度都要相同，实际上也只是最后两个维度相乘。
            #以软分配矩阵的概率值做加权平均
        nodes = F.normalize(nodes, p=2, dim=2) #intra-normalization，对通道C维度除以通道维度的二范数进行规范化
        #normalize(nodes, p, dim)  p取值可以是1/2，分别表示L1/L2范数   对第dim维度操作，可以把矩阵想成一个立方体，三个维度分别是前后0，左右1，上下2
        nodes = nodes.view(B, -1).contiguous() # B X (Node_num X C)
        nodes = F.normalize(nodes, p=2, dim=1) # l2 normalize   #将1 2维合并又规范了一次
        return nodes.view(B, self.node_num, C).permute(0,2,1).contiguous(), soft_assign
        # B*C*node，B * node * n





class GraphProcess(nn.Module):
    def __init__(self, bnum, bnod, dim, loop, thr4, thr2, thr1):
        super(GraphProcess, self).__init__()
        self.loop = loop
        self.bnum = bnum
        self.bnod = bnod
        self.dim = dim
        self.node_num = bnum*bnum*bnod
        self.proj = GraphProjection(self.bnum, self.bnod, self.dim)
        self.gconv = GraphInference(self.dim, self.loop, self.bnum, thr4, thr2, thr1)

    def GraphReprojection(self,Q,Z):
        self.B, self.Dim, _ = Z.size()
        res=torch.zeros([self.B,self.Dim,self.H,self.W], device=Z.device, dtype=Z.dtype, layout=Z.layout)
        for node_id in range(self.node_num):
            block_id=math.floor(node_id/self.bnod)
            h_sta=math.floor(block_id/self.bnum)*self.h
            w_sta=block_id%(self.bnum)*self.w
            h_end=h_sta+self.h
            w_end=w_sta+self.w
            # 可优化
            res[:,:, h_sta:h_end , w_sta:w_end]+=torch.matmul(Z[:,:,node_id].unsqueeze(2),Q[:,node_id,:].unsqueeze(1)).view(self.B,self.Dim,self.h,self.w)
        return res  #return torch.matmul(Q.permute(0,2,1).contiguous(),Z.permute(0,2,1).contiguous())

    def forward(self, x):  #
        _, _, self.H, self.W = x.size()
        self.h = math.floor(self.H / self.bnum)
        self.w = math.floor(self.W / self.bnum)
        g, Q = self.proj(x)
        g = self.gconv(g)
        res = self.GraphReprojection(Q, g) + x
        res.to(x.device)
        return res  # g_cgi is B C node_num




class DSIModule(nn.Module):
    def __init__(self,channel):
        super(DSIModule, self).__init__()
        self.dim=1280
        self.gcnlayer=GraphProcess(bnum=7, bnod=2, dim=512, loop=3, thr4=0.2, thr2=0.4, thr1=0.6)

        self.conv=BasicConv2d(self.dim, 512 ,kernel_size=1,stride=1)

        self.downsample2 = nn.MaxPool2d(2,stride=2)
        self.downsample4 = nn.MaxPool2d(4,stride=4)
        self.downsample8 = nn.MaxPool2d(8,stride=8)

    def forward(self,x1,x2,x3,x4,x5):
        x3 = self.downsample8 (x3)
        x4 = self.downsample4 (x4)
        x5 = self.downsample2 (x5)
        x=torch.cat((x3,x4,x5),dim=1)
        x=self.conv(x)
        x=self.gcnlayer(x)+x
        return x