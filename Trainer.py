import logging

import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_data_loader, val_data_loader, n_epochs, lr,
                 save_epoch_int, model_folder, device):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.n_epochs = n_epochs
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
        batch_x, batch_n, batch_y, batch_x_len, batch_n_len, batch_y_len, batch_traj_poi_pos, batch_traj_poi_neg, \
        poi_pos, poi_neg = data
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_n = batch_n.to(self.device)
        poi_pos = poi_pos.to(self.device)
        poi_neg = poi_neg.to(self.device)
        batch_traj_poi_pos = batch_traj_poi_pos.to(self.device)
        batch_traj_poi_neg = batch_traj_poi_neg.to(self.device)

        last_output_emb, poi_emb, traj_poi_emb = self.model((batch_x, batch_x_len,
                                                             None,
                                                             None,
                                                             ), True)

        last_output_y_emb, pos_poi_emb, traj_poi_emb_pos = self.model((batch_y, batch_y_len,
                                                                       poi_pos,
                                                                       batch_traj_poi_pos,
                                                                       ), False)

        last_output_n_emb, n_poi_emb, traj_poi_emb_neg = self.model((batch_n, batch_n_len,
                                                                     poi_neg,
                                                                     batch_traj_poi_neg
                                                                     ), False)

        prediction_i = (last_output_emb * last_output_y_emb).sum(dim=-1)
        prediction_j = (last_output_emb * last_output_n_emb).sum(dim=-1)

        loss1 = - (prediction_i - prediction_j).sigmoid().log().sum()
        loss2 = -((poi_emb * pos_poi_emb).sum(dim=-1) -
                  (poi_emb * n_poi_emb).sum(dim=-1)).sigmoid().log().sum()
        loss3 = -((traj_poi_emb * traj_poi_emb_pos).sum(dim=-1) -
                  (traj_poi_emb * traj_poi_emb_neg).sum(dim=-1)).sigmoid().log().sum()
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
            pbar.set_description('[loss: %f]' % loss)

        return np.array(losses).mean()

    def _val_epoch(self):
        self.model.eval()

        losses = []

        pbar = tqdm(self.val_data_loader)
        for data in pbar:
            loss = self._pass(data, train=False)
            losses.append(loss)
            pbar.set_description('[loss: %f]' % loss)

        return np.array(losses).mean()

    def train(self):
        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch()
            logging.info(
                '[Epoch %d/%d] [training loss: %f]' %
                (epoch, self.n_epochs, train_loss)
            )

            if (epoch + 1) % self.save_epoch_int == 0:
                save_file = self.model_folder + '/epoch_%d.pt' % epoch
                torch.save(self.model.state_dict(), save_file)
