import numpy as np
import torch
import random
import os

import logging
from logg import setup_logger
from pars_args import args
from LSTM_Model import LSTM_Tra
from Data_Loader import load_traindata
from Data_Loader import load_testdata
from Data_Loader import TestValueDataLoader
from Data_Loader import TrainDataValLoader

from gcn_emb import gcn_emb


def recall(s_emb, train_emb, label, K=[1, 5, 10, 20, 50]):
    r = np.dot(s_emb, train_emb.T)
    label_r = np.argsort(-r, axis=1)
    recall = np.zeros((s_emb.shape[0], len(K)))
    for idx, la in enumerate(label):
        for idx_k, k in enumerate(K):
            if la in label_r[idx, :k]:
                recall[idx, idx_k:] = 1
                break
    return recall


def eval_model():
    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

    emb_weights = gcn_emb()
    train_x, train_y, _ = load_traindata(args.train_file)
    val_x, val_y = load_testdata(args.test_file)

    emb_train = np.zeros((len(train_x), args.latent_dim))

    model = LSTM_Tra(args.n_layers, args.latent_dim, 0.5, device, emb_weights, batch_first=True).to(device)

    for epoch in range(199, 200):
        model_name = 'epoch_' + str(epoch) + '.pt'

        # model_f = '%s/%s' % (args.save_gcn_folder + '_' + str(args.alpha), model_name)
        model_f = 'saved_models_0.5_GAT/' + model_name
        if not os.path.exists(model_f):
            continue

        logging.info('Loading value nn from %s' % model_f)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        data_loader_val, poi_label_val, poi_neg_val = TestValueDataLoader(args.test_file, emb_weights,
                                                                          args.poi_file, args.batch_size)

        data_loader_train, poi_label_train, poi_neg_train = TrainDataValLoader(args.train_file, emb_weights,
                                                                               args.poi_file, args.batch_size)

        print('111')
        for batch_id, (batch_x, batch_y, batch_x_len, idx_list) in enumerate(data_loader_train):
            batch_x = batch_x.to(device)
            last_output_emb, _ = model((batch_x, batch_x_len,
                                        poi_label_train.to(device), poi_neg_train.to(device)), True)

            emb_train[idx_list, :] = last_output_emb.cpu().detach().numpy()

        K = [1, 5, 10, 20, 50]


        print('222')
        rec = np.zeros((val_x.shape[0], len(K)))
        for batch_id, (batch_x, batch_y, batch_x_len, idx_list) in enumerate(data_loader_val):
            batch_x = batch_x.to(device)
            last_output_emb, _ = model((batch_x, batch_x_len,
                                        poi_label_val.to(device), poi_neg_val.to(device)), True)
            rec[idx_list, :] = recall(last_output_emb.cpu().detach().numpy(), emb_train, batch_y, K)
        print('333')

        rec_ave = rec.mean(axis=0)
        for rec in rec_ave:
            logging.info('%.4f' % rec)

        print(rec_ave)


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('LSTM_gcn_test_%.2f.log' % args.alpha)

    eval_model()
