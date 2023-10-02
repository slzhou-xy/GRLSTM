import numpy as np
import scipy.sparse as sp
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--nodes',          type=int,   default=28342,
                    help='Newyork=95581, Beijing=28342')
parser.add_argument('--datapath',        type=str,   default='../data/',
                    help='bj=beijing, ny=newyork')
parser.add_argument('--dataset',        type=str,   default='bj',
                    help='bj=beijing, ny=newyork')
args = parser.parse_args()


def data_clean():
    data = np.loadtxt(args.datapath + args.dataset + '_e.txt', dtype=int)
    data = np.unique(data, axis=0)
    np.savetxt(args.datapath + args.dataset + '_e_unique.txt', data, fmt='%d %d')


def construct_e_map():
    data = np.arange(args.nodes).reshape((args.nodes, 1))
    data = np.concatenate((data, data), axis=1)
    np.savetxt(args.datapath + args.dataset + '_e_map.txt', data, fmt='%d %d')


def tra_directed_edge():
    tra_list = np.load(args.datapath + args.dataset + '_tra.npy', allow_pickle=True)
    T = sp.lil_matrix((args.nodes, args.nodes), dtype=bool)
    for idx, tra in enumerate(tra_list):
        if idx % 10000 == 0:
            print(idx)
        left = 0
        right = left + 1
        while right < len(tra):
            T[tra[left], tra[right]] = True
            left += 1
            right += 1
    sp.save_npz(args.datapath + args.dataset + '_tra_edge.npz', T.tocsr())


def find_directed_neighbors():
    neighbors = [[] for _ in range(args.nodes)]
    e = np.loadtxt(args.datapath + args.dataset + '_e_unique.txt', dtype=int)
    for edge in e:
        neighbors[edge[0]].append(edge[1])
    # np.savez(args.datapath + args.dataset + '_poi_neighbors.npz', neighbors=neighbors)
    pickle.dump(neighbors, open(args.datapath + args.dataset + '_poi_neighbors.pkl', 'wb')


def contruct_directed_knowledge_graph():
    # neighbors = np.load(args.datapath + args.dataset + '_poi_neighbors.npz', allow_pickle=True)['neighbors']
    neighbors = pickle.load(open(args.datapath + args.dataset + '_poi_neighbors.pkl', 'rb'))
    lens = 0
    for n in neighbors:
        lens += len(n)
    print('lens', lens)

    only_road_network = sp.lil_matrix((args.nodes, args.nodes), dtype=bool)
    traj_edge_in_road_network = sp.lil_matrix((args.nodes, args.nodes), dtype=bool)
    traj_edge_not_in_road_network = sp.lil_matrix((args.nodes, args.nodes), dtype=bool)

    all_traj_edge = sp.coo_matrix(sp.load_npz(args.datapath + args.dataset + '_tra_edge.npz'))
    all_traj_edge = np.vstack((all_traj_edge.row, all_traj_edge.col)).T

    print('all_traj_edge', all_traj_edge.shape)

    for edge in all_traj_edge:
        if edge[1] in neighbors[edge[0]]:
            traj_edge_in_road_network[edge[0], edge[1]] = True
        if edge[1] not in neighbors[edge[0]]:
            traj_edge_not_in_road_network[edge[0], edge[1]] = True

    unique_edge = np.loadtxt(args.datapath + args.dataset + '_e_unique.txt', dtype=int)  # not change

    i = 0
    j = 0

    for edge in unique_edge:
        if not traj_edge_in_road_network[edge[0], edge[1]]:
            i += 1
            only_road_network[edge[0], edge[1]] = True
        else:
            j += 1

    print(i, j)

    only_road_network = only_road_network.tocoo()
    only_road_network = np.vstack((only_road_network.row, only_road_network.col)).T
    print('only_road_network', only_road_network.shape)

    beijing_edge_type = np.zeros((only_road_network.shape[0], 1), dtype=int)
    beijing_with_edge_type = np.concatenate((only_road_network, beijing_edge_type), axis=1)

    traj_edge_in_road_network = traj_edge_in_road_network.tocoo()
    traj_edge_in_road_network = np.vstack((traj_edge_in_road_network.row, traj_edge_in_road_network.col)).T

    print('traj_edge_in_road_network', traj_edge_in_road_network.shape)

    traj_edge_in_road_network_edge_type = np.ones((traj_edge_in_road_network.shape[0], 1), dtype=int)
    traj_edge_in_road_network_edge_type = np.concatenate((traj_edge_in_road_network,
                                                          traj_edge_in_road_network_edge_type), axis=1)

    traj_edge_not_in_road_network = traj_edge_not_in_road_network.tocoo()
    traj_edge_not_in_road_network = np.vstack((traj_edge_not_in_road_network.row, traj_edge_not_in_road_network.col)).T
    print('traj_edge_not_in_road_network', traj_edge_not_in_road_network.shape)

    traj_edge_not_in_road_network_edge_type = np.ones((traj_edge_not_in_road_network.shape[0], 1), dtype=int) + 1
    traj_edge_not_in_road_network_edge_type = np.concatenate((traj_edge_not_in_road_network,
                                                              traj_edge_not_in_road_network_edge_type), axis=1)

    fusion_graph = np.concatenate((beijing_with_edge_type, traj_edge_in_road_network_edge_type,
                                   traj_edge_not_in_road_network_edge_type), axis=0)
    print('fusion', fusion_graph.shape)

    np.savetxt(args.datapath + args.dataset + '_KG_graph.txt', fusion_graph, fmt="%d %d %d")


if __name__ == '__main__':
    data_clean()
    construct_e_map()
    tra_directed_edge()
    find_directed_neighbors()
    contruct_directed_knowledge_graph()
