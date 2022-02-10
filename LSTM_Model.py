import torch
import torch.nn as nn
import logging
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


class LSTM_Tra(nn.Module):

    def __init__(self, n_layers, latent_dim, dropout_rate, device, emb_weights, batch_first=True):
        super(LSTM_Tra, self).__init__()
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.device = device
        self.batch_first = batch_first

        logging.info('Initializing model: latent_dim=%d' % self.latent_dim)

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(emb_weights), freeze=False)
        # self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(emb_weights))

        # self.GCN = nn.Linear(emb_weights.shape[1], latent_dim)

        self.LSTM1 = nn.LSTM(input_size=latent_dim, hidden_size=latent_dim, num_layers=n_layers,
                             batch_first=batch_first)
        self.LSTM2 = nn.LSTM(input_size=latent_dim, hidden_size=latent_dim, num_layers=n_layers,
                             batch_first=batch_first)
        self.LSTM3 = nn.LSTM(input_size=latent_dim, hidden_size=latent_dim, num_layers=n_layers,
                             batch_first=batch_first)
        self.relu = nn.ReLU()

    def forward(self, fps, pos=True):
        if pos:
            # batch_x为一个批次的所有轨迹，一个批次为128
            # batch_x_len为这个批次所有轨迹的长度，长度之前已经对齐
            batch_x, batch_x_len, poi_label, poi_neg = fps
            # batch_x(128, 这128条轨迹的长度)
            batch_emb = self.embedding(batch_x)
            # batch_emb(128, 这128条轨迹的长度, 每个poi变成128维embedding)

            # batch_emb = self.GCN(batch_emb)
            # 得到GCN的embedding，维度不变

            # pack_padded_sequence输入必须按照长度排序，长度在前面
            # input：要压缩的数据。当batch_first是True时候，shape的输入格式是[B,S,*]，
            # 其中B是batch_size,S是seq_len(该batch中最长序列的长度)，*可以是任何维度。
            # 如果batch_first是True时候，相应的的数据格式必须是[S,B,*]
            batch_emb_pack = rnn_utils.pack_padded_sequence(batch_emb, batch_x_len, batch_first=self.batch_first)
            # batch_emb_pack(batch_x_len的总和, 128)
            # batch_emb_pack.data.shape(当前这一批次batch_x_len的总和, 128)
            # batch_emb_pack.batch_sizes为batch_x_len的一维tensor

            # out_emb和batch_emb_pack格式相同
            out_emb, _ = self.LSTM1(batch_emb_pack)

            # 变回成原来状态
            out_emb_pad, out_emb_len = rnn_utils.pad_packed_sequence(out_emb, batch_first=self.batch_first)

            out_emb_pad = batch_emb + self.relu(out_emb_pad)
            # out_emb_pad += batch_emb
            # out_emb_pad = self.relu(out_emb_pad)

            batch_emb_pack = rnn_utils.pack_padded_sequence(out_emb_pad, out_emb_len, batch_first=self.batch_first)

            out_emb, _ = self.LSTM2(batch_emb_pack)

            out_emb_pad, out_emb_len = rnn_utils.pad_packed_sequence(out_emb, batch_first=self.batch_first)

            out_emb_pad = batch_emb + self.relu(out_emb_pad)
            # out_emb_pad += batch_emb
            # out_emb_pad = self.relu(out_emb_pad)

            batch_emb_pack = rnn_utils.pack_padded_sequence(out_emb_pad, out_emb_len, batch_first=self.batch_first)

            out_emb, _ = self.LSTM3(batch_emb_pack)

            out_emb_pad, out_emb_len = rnn_utils.pad_packed_sequence(out_emb, batch_first=self.batch_first)

            # print(out_emb_pad.shape)
            # out_n_emb_pad(批次, 这批次的轨迹长度, embedding维度)
            # out_emb_len = batch_x_len

            idx = (torch.LongTensor(batch_x_len) - 1).view(-1, 1).expand(
                len(batch_x_len), out_emb_pad.size(2))
            time_dimension = 1 if self.batch_first else 0
            idx = idx.unsqueeze(time_dimension)
            if out_emb_pad.is_cuda:
                idx = idx.cuda(out_emb_pad.data.get_device())
            # Shape: (batch_size, rnn_hidden_dim)
            last_output_emb = out_emb_pad.gather(
                time_dimension, Variable(idx)).squeeze(time_dimension)

            poi_prediction_i = torch.zeros((batch_x.shape[1])).to(self.device)
            poi_prediction_j = torch.zeros((batch_x.shape[1])).to(self.device)

            for i in range(batch_x.shape[0]):
                label_index = poi_label.index_select(0, batch_x[i])
                neg_index = poi_neg.index_select(0, batch_x[i])
                # label_index = poi_label(batch_x[i])
                # neg_index = poi_neg(batch_x[i])
                poi_prediction_i += (self.embedding(batch_x[i]) *
                                     self.embedding(label_index)).sum(-1) / batch_x.shape[1]
                poi_prediction_j += (self.embedding(batch_x[i]) *
                                     self.embedding(neg_index)).sum(-1) / batch_x.shape[1]

        else:
            batch_n, batch_n_len, poi_label, poi_neg = fps
            # print(batch_n.shape)
            batch_n_emb = self.embedding(batch_n)

            # batch_n_emb = self.GCN(batch_n_emb)

            sorted_seq_lengths, indices = torch.sort(torch.IntTensor(batch_n_len), descending=True)
            batch_n_emb = batch_n_emb[indices]
            _, desorted_indices = torch.sort(indices, descending=False)

            batch_emb_n_pack = rnn_utils.pack_padded_sequence(batch_n_emb, sorted_seq_lengths,
                                                              batch_first=self.batch_first)

            out_n_emb, _ = self.LSTM1(batch_emb_n_pack)

            out_n_emb_pad, out_n_emb_len = rnn_utils.pad_packed_sequence(out_n_emb, batch_first=self.batch_first)

            out_n_emb_pad = batch_n_emb + self.relu(out_n_emb_pad)

            # out_n_emb_pad += batch_n_emb
            # out_n_emb_pad = self.relu(out_n_emb_pad)

            batch_emb_n_pack = rnn_utils.pack_padded_sequence(out_n_emb_pad, out_n_emb_len,
                                                              batch_first=self.batch_first)

            out_n_emb, _ = self.LSTM2(batch_emb_n_pack)

            out_n_emb_pad, out_n_emb_len = rnn_utils.pad_packed_sequence(out_n_emb, batch_first=self.batch_first)

            out_n_emb_pad = batch_n_emb + self.relu(out_n_emb_pad)

            # out_n_emb_pad += batch_n_emb
            # out_n_emb_pad = self.relu(out_n_emb_pad)

            batch_emb_n_pack = rnn_utils.pack_padded_sequence(out_n_emb_pad, out_n_emb_len,
                                                              batch_first=self.batch_first)

            out_n_emb, _ = self.LSTM3(batch_emb_n_pack)

            out_n_emb_pad, out_n_emb_len = rnn_utils.pad_packed_sequence(out_n_emb, batch_first=self.batch_first)

            out_n_emb_pad = out_n_emb_pad[desorted_indices]

            idx = (torch.LongTensor(batch_n_len) - 1).view(-1, 1).expand(
                len(batch_n_len), out_n_emb_pad.size(2))
            time_dimension = 1 if self.batch_first else 0
            idx = idx.unsqueeze(time_dimension)
            if out_n_emb_pad.is_cuda:
                idx = idx.cuda(out_n_emb_pad.data.get_device())
            # Shape: (batch_size, rnn_hidden_dim)
            last_output_emb = out_n_emb_pad.gather(
                time_dimension, Variable(idx)).squeeze(time_dimension)

            # poi传播
            poi_prediction_i = torch.zeros((batch_n.shape[1])).to(self.device)
            poi_prediction_j = torch.zeros((batch_n.shape[1])).to(self.device)

            for i in range(batch_n.shape[0]):
                label_index = poi_label.index_select(0, batch_n[i])
                neg_index = poi_neg.index_select(0, batch_n[i])
                # label_index = poi_label(batch_n[i])
                # neg_index = poi_neg(batch_n[i])
                poi_prediction_i += (self.embedding(batch_n[i]) *
                                     self.embedding(label_index)).sum(-1) / batch_n.shape[1]
                poi_prediction_j += (self.embedding(batch_n[i]) *
                                     self.embedding(neg_index)).sum(-1) / batch_n.shape[1]

        out_put_poi = (poi_prediction_i - poi_prediction_j) / 3

        return last_output_emb, out_put_poi
