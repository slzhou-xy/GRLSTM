import numpy as np
from pars_args import args
from scipy import sparse
from scipy.sparse import lil_matrix


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


def normalize_adj(adj, alpha=1.):
    adj = sparse.eye(adj.shape[0]) + alpha * adj
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocsr()


def generate_graph(alpha, graph_file, len_n):
    graph = lil_matrix((len_n, len_n))
    edge_list = np.loadtxt(graph_file, dtype=int)
    for edge in edge_list:
        graph[edge[0], edge[1]] = 1
        graph[edge[1], edge[0]] = 1

    return normalize_adj(graph.tocsr(), alpha)


# def gcn_emb():
#     emb = load_weight(args.emb_file)
#     len_n = emb.shape[0]
#     A = generate_graph(args.alpha, args.graph_file, len_n)
#     emb[:len_n, :] = A.dot(emb[:len_n, :])
#
#     return emb
def gcn_emb():
    emb = np.loadtxt(r'E:\PyCharm_Project\OTRA_train_with_GAT\gat_emb128.txt')
    return emb