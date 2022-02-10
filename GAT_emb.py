import numpy as np
import torch
import torch_geometric.data.data
from scipy import sparse
from torch_geometric.nn import GATConv
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader


def load_weight(emb_file):
    weights_l = np.loadtxt(emb_file, skiprows=1)  # 跳过第一行
    # weights_l (28343, 129)
    # 因为文件每一行第一个为node节点号
    emb_w = np.zeros((weights_l.shape[0], weights_l.shape[1] - 1))
    # emb_w (28342, 128)
    print(emb_w.shape)
    for weight in weights_l:  # 选择每一行
        # idx = int(weight[0])  # 选择节点号
        idx = int(weight[0]) - 1  # 其他维度需要减1
        emb_w[idx, :] = weight[1:]  # 在emb中重新放入对应顺序
    return emb_w


def construct_data():
    emb = load_weight(r'E:\PyCharm_Project\OTRA_train_with_GAT\beijing.emb')
    graph = np.eye(emb.shape[0])
    edge_list = np.loadtxt(r'E:\PyCharm_Project\OTRA_train_with_GAT\beijing.edgelist', dtype=int)
    edge_list = np.sort(edge_list, axis=0)
    for edge in edge_list:
        graph[edge[0], edge[1]] = 1
        graph[edge[1], edge[0]] = 1
    edge_index = sparse.coo_matrix(graph)
    edge_index = np.vstack((edge_index.row, edge_index.col))
    return emb, edge_index


class GATLayer(nn.Module):
    def __init__(self):
        super(GATLayer, self).__init__()

        self.GAT = GATConv(in_channels=128, out_channels=512, heads=8, dropout=0.1, concat=True)
        self.lin = nn.Linear(512 * 8, 512)

    def forward(self, x, edge_index):
        return self.lin(self.GAT(x, edge_index))


if __name__ == '__main__':
    emb, edge_index = construct_data()
    model = GATLayer().to('cuda')
    emb = torch.Tensor(emb).to('cuda')
    edge_index = torch.LongTensor(edge_index).to('cuda')
    emb = model(emb, edge_index)
    emb = emb.cpu().detach().numpy()
    np.savetxt('gat_emb512.txt', emb)
