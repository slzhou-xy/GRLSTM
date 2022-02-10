import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import logging


class Trainer:
    def __init__(self, model, train_data_loader, val_data_loader, poi_label, poi_neg,
                 n_epochs, alpha, lr,
                 save_epoch_int, model_folder, device):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.poi_label = poi_label.to(device)
        self.poi_neg = poi_neg.to(device)
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.lr = lr
        self.save_epoch_int = save_epoch_int
        self.model_folder = model_folder
        self.device = device
        self.model = model.to(self.device)

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

    def _pass(self, data, train=True):

        self.optim.zero_grad()

        batch_x, batch_n, batch_y, batch_x_len, batch_n_len, batch_y_len, \
        double_neighbor_neg, double_neighbor_data_label, double_neighbor_neg_length, double_neighbor_label_length = data

        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_n = batch_n.to(self.device)
        self.poi_label = self.poi_label.to(self.device)
        self.poi_neg = self.poi_neg.to(self.device)
        double_neighbor_neg = double_neighbor_neg.to(self.device)
        double_neighbor_data_label = double_neighbor_data_label.to(self.device)

        # t_i
        last_output_emb, output_poi = self.model((batch_x, batch_x_len,
                                                  self.poi_label,
                                                  self.poi_neg
                                                  ), True)

        # t_i'
        last_output_y_emb, y_output_poi = self.model((batch_y, batch_y_len,
                                                      self.poi_label,
                                                      self.poi_neg
                                                      ), False)
        # t_j
        last_output_n_emb, n_output_poi = self.model((batch_n, batch_n_len,
                                                      self.poi_label,
                                                      self.poi_neg
                                                      ), False)

        # double_neighbor_label
        last_output_neighbor_y_emb, _ = self.model((double_neighbor_data_label,
                                                    double_neighbor_label_length,
                                                    self.poi_label,
                                                    self.poi_neg), False)
        # double_neighbor_neg
        last_output_neighbor_n_emb, _ = self.model((double_neighbor_neg,
                                                    double_neighbor_neg_length,
                                                    self.poi_label,
                                                    self.poi_neg), False)

        neighbor_prediction_i = (last_output_emb * last_output_neighbor_y_emb).sum(dim=-1)
        neighbor_prediction_j = (last_output_emb * last_output_neighbor_n_emb).sum(dim=-1)

        # last_output_n_emb(128, 128)

        # 行求和，再变成一行
        # last_output_n_emb * last_output_y_emb对应元素求和，再sum，就为对每行求点积
        prediction_i = (last_output_emb * last_output_y_emb).sum(dim=-1)
        prediction_j = (last_output_emb * last_output_n_emb).sum(dim=-1)

        # 公式12
        loss1 = - (prediction_i - prediction_j).sigmoid().log().sum()

        '''
        ADD
        '''
        loss2 = - (output_poi.sigmoid().log().sum() +
                   y_output_poi.sigmoid().log().sum() +
                   n_output_poi.sigmoid().log().sum())

        loss3 = -(neighbor_prediction_i - neighbor_prediction_j).sigmoid().log().sum()

        loss = loss1 + loss2 + loss3

        if train:
            torch.backends.cudnn.enabled = False
            loss.backward()
            self.optim.step()

        return loss.item()

    def _train_epoch(self):
        self.model.train()

        losses = []
        pbar = tqdm(self.train_data_loader)
        for data in pbar:
            loss = self._pass(data)
            losses.append(loss)
            pbar.set_description('[loss: %f]' % (loss))

        return np.array(losses).mean()

    def _val_epoch(self):
        self.model.eval()

        losses = []

        pbar = tqdm(self.val_data_loader)
        for data in pbar:
            loss = self._pass(data, train=False)
            losses.append(loss)
            pbar.set_description('[loss: %f]' % (loss))

        return np.array(losses).mean()

    def train(self):
        # best_val_loss = np.inf
        for epoch in range(self.n_epochs):
            # self.train_data_loader.reshuffle()

            train_loss = self._train_epoch()
            # val_loss = self._val_epoch()
            logging.info(
                '[Epoch %d/%d] [training loss: %f] [validation loss: %f]' %
                (epoch, self.n_epochs, train_loss, 0)
            )

            '''if val_loss < best_val_loss or epoch==self.n_epochs-1:
                best_val_loss = val_loss
                save_file = self.model_folder + '/best_epoch_%d.pt' % epoch
                torch.save(self.model.state_dict(), save_file)'''

            if (epoch + 1) % self.save_epoch_int == 0:
                save_file = self.model_folder + '/epoch_%d.pt' % epoch
                torch.save(self.model.state_dict(), save_file)
