import torch
import torch.nn as nn
import logging
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from pars_args import args
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import numpy as np


class GRLSTM(nn.Module):
    def __init__(self, args, device, batch_first=True):
        super(GRLSTM, self).__init__()
        self.nodes = args.nodes
        self.latent_dim = args.latent_dim
        self.device = device
        self.batch_first = batch_first
        self.num_heads = args.num_heads

        logging.info('Initializing model: latent_dim=%d' % self.latent_dim)

        self.poi_neighbors = np.load(
            args.poi_file, allow_pickle=True)['neighbors']

        # self.poi_features = torch.randn(
        #     self.nodes, self.latent_dim).to(self.device)
        if not os.path.exists('data/poi_features.npy'):
            poi_features = np.random.randn(self.nodes, self.latent_dim)
            np.save('data/poi_features.npy', poi_features)
        else:
            poi_features = np.load('data/poi_features.npy')
        self.poi_features = torch.tensor(poi_features, dtype=torch.float32).to(self.device)

        self.lstm_list = nn.ModuleList([
            nn.LSTM(input_size=self.latent_dim, hidden_size=self.latent_dim,
                    num_layers=1, batch_first=True)
            for _ in range(args.lstm_layers)
        ])

        self.gat = GATConv(in_channels=self.latent_dim, out_channels=16,
                           heads=8, dropout=0.1, concat=True)

    def _construct_edge_index(self, batch_x_flatten):
        batch_x_flatten = batch_x_flatten.cpu().numpy()
        neighbors = self.poi_neighbors[batch_x_flatten]

        batch_x_flatten = batch_x_flatten.repeat(neighbors.shape[1])

        neighbors = neighbors.reshape(-1)

        edge_index = np.vstack((neighbors, batch_x_flatten))
        batch_x_flatten = np.vstack((batch_x_flatten, batch_x_flatten))
        edge_index = np.concatenate((batch_x_flatten, edge_index), axis=1)
        edge_index = np.unique(edge_index, axis=1)

        return torch.tensor(edge_index).to(self.device)

    def forward(self, fps, pos=True):
        if pos:
            batch_x, batch_x_len, _, _ = fps
            batch_x_flatten = batch_x.reshape(-1)
            batch_x_flatten = torch.unique(batch_x_flatten)
            batch_edge_index = self._construct_edge_index(batch_x_flatten)
            embedding_weight = self.gat(self.poi_features, batch_edge_index)
            batch_emb = embedding_weight[batch_x]

            batch_emb_pack = rnn_utils.pack_padded_sequence(
                batch_emb, batch_x_len, batch_first=self.batch_first)

            for lstm in self.lstm_list[:-1]:
                out_emb, _ = lstm(batch_emb_pack)
                out_emb_pad, out_emb_len = rnn_utils.pad_packed_sequence(
                    out_emb, batch_first=self.batch_first)
                out_emb_pad = batch_emb + F.relu(out_emb_pad)
                batch_emb_pack = rnn_utils.pack_padded_sequence(
                    out_emb_pad, out_emb_len, batch_first=self.batch_first)
            out_emb, _ = self.lstm_list[-1](batch_emb_pack)
            out_emb_pad, out_emb_len = rnn_utils.pad_packed_sequence(
                out_emb, batch_first=self.batch_first)

            idx = (torch.LongTensor(batch_x_len) - 1).view(-1, 1).expand(
                len(batch_x_len), out_emb_pad.size(2))
            time_dimension = 1 if self.batch_first else 0
            idx = idx.unsqueeze(time_dimension)
            if out_emb_pad.is_cuda:
                idx = idx.cuda(out_emb_pad.data.get_device())
            last_output_emb = out_emb_pad.gather(
                time_dimension, Variable(idx)).squeeze(time_dimension)

            traj_poi_emb = batch_emb
            poi_emb = batch_emb
        else:
            batch_n, batch_n_len, poi, batch_traj_poi = fps
            batch_n_flatten = batch_n.reshape(-1)
            batch_n_flatten = torch.unique(batch_n_flatten)
            batch_n_edge_index = self._construct_edge_index(batch_n_flatten)
            embedding_weight = self.gat(self.poi_features, batch_n_edge_index)
            batch_n_emb = embedding_weight[batch_n]

            sorted_seq_lengths, indices = torch.sort(
                torch.IntTensor(batch_n_len), descending=True)
            batch_n_emb = batch_n_emb[indices]
            _, desorted_indices = torch.sort(indices, descending=False)

            batch_emb_n_pack = rnn_utils.pack_padded_sequence(batch_n_emb,
                                                              sorted_seq_lengths,
                                                              batch_first=self.batch_first)

            for lstm in self.lstm_list[:-1]:
                out_n_emb, _ = lstm(batch_emb_n_pack)
                out_n_emb_pad, out_n_emb_len = rnn_utils.pad_packed_sequence(
                    out_n_emb, batch_first=self.batch_first)
                out_n_emb_pad = batch_n_emb + F.relu(out_n_emb_pad)
                batch_emb_n_pack = rnn_utils.pack_padded_sequence(out_n_emb_pad,
                                                                  out_n_emb_len,
                                                                  batch_first=self.batch_first)
            out_n_emb, _ = self.lstm_list[-1](batch_emb_n_pack)
            out_n_emb_pad, out_n_emb_len = rnn_utils.pad_packed_sequence(
                out_n_emb, batch_first=self.batch_first)

            out_n_emb_pad = out_n_emb_pad[desorted_indices]
            idx = (torch.LongTensor(batch_n_len) - 1).view(-1, 1).expand(
                len(batch_n_len), out_n_emb_pad.size(2))
            time_dimension = 1 if self.batch_first else 0
            idx = idx.unsqueeze(time_dimension)
            if out_n_emb_pad.is_cuda:
                idx = idx.cuda(out_n_emb_pad.data.get_device())
            last_output_emb = out_n_emb_pad.gather(
                time_dimension, Variable(idx)).squeeze(time_dimension)

            if batch_traj_poi is not None:
                batch_traj_poi_flatten = batch_traj_poi.reshape(-1)
                batch_traj_poi_flatten = torch.unique(batch_traj_poi_flatten)
                batch_traj_poi_edge_index = self._construct_edge_index(
                    batch_traj_poi_flatten)
                embedding_weight = self.gat(
                    self.poi_features, batch_traj_poi_edge_index)
                traj_poi_emb = embedding_weight[batch_traj_poi]
            else:
                traj_poi_emb = None

            if poi is not None:
                poi_flatten = poi.reshape(-1)
                poi_flatten = torch.unique(poi_flatten)
                poi_edge_index = self._construct_edge_index(poi_flatten)
                embedding_weight = self.gat(self.poi_features, poi_edge_index)
                poi_emb = embedding_weight[poi]
            else:
                poi_emb = None

        return last_output_emb, poi_emb, traj_poi_emb
