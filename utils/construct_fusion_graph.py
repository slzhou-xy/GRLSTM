import torch
import numpy as np
import scipy.sparse as sp
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--nodes',          type=int,   default=28342,
                    help='Newyork=95581, Beijing=28342')
parser.add_argument('--datapath',        type=str,   default='../data/',
                    help='bj=beijing, ny=newyork')
parser.add_argument('--dataset',        type=str,   default='bj',
                    help='bj=beijing, ny=newyork')
parser.add_argument('--transhpath',        type=str,   default='../KGE/log/beijing-transh.ckpt_final',)
parser.add_argument('--k',        type=int,   default=10,
                    help='bj=10, ny=30')
args = parser.parse_args()


if __name__ == '__main__':
    emb = torch.load(args.transhpath)
    poi_emb = emb['model_state_dict']['ent_embeddings.weight'].cpu().numpy()
    rel_emb = emb['model_state_dict']['rel_embeddings.weight'].cpu().numpy()
    KG = np.loadtxt(args.datapath + args.dataset + '_KG_graph.txt', dtype=int)

    M = np.zeros((args.nodes, args.nodes))
    
    for i in range(args.nodes):
        M[i] = np.exp(-np.linalg.norm(poi_emb[i] - poi_emb, ord=2, axis=1))

    for edge in KG:
        edge_type = edge[2]
        start = edge[0]
        end = edge[1]
        l2 = np.exp(-np.linalg.norm(poi_emb[start] + rel_emb[edge_type] - poi_emb[end], ord=2))
        M[start, end] = l2

    edge = np.argsort(M)[:, -args.k - 1:-1]
    print(edge.shape)
    poi = np.arange(M.shape[0])
    del M
    np.savez(args.datapath + args.dataset + 'transh_poi_10', poi=poi, neighbors=edge)
    print("over")
