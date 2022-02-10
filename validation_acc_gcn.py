import numpy as np
import torch
import random
import os

import logging
from logg import setup_logger
from pars_args import args
from LSTM_Model import LSTM_Tra
from tqdm import tqdm
from Data_Loader import load_traindata
from Data_Loader import load_valdata
from Data_Loader import ValValueDataLoader
from Data_Loader import TrainDataValLoader
from gcn_emb import gcn_emb


def recall(s_emb, train_emb, label, device, K=[1, 5, 10, 20, 50]):
    r = torch.mm(s_emb, torch.transpose(train_emb, 0, 1))
    label_r = torch.argsort(r, dim=1, descending=True)
    recall = torch.zeros((s_emb.shape[0], len(K)), device=device)
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
    val_x, val_y = load_valdata(args.val_file)

    emb_train = torch.zeros((len(train_x), args.latent_dim), device=device, requires_grad=False)

    K = [1, 5, 10, 20, 50]
    model = LSTM_Tra(args.n_layers, args.latent_dim, 0.5, device, emb_weights, batch_first=True).to(device)

    rec = torch.zeros((val_x.shape[0], len(K)), device=device, requires_grad=False)
    data_loader_train, poi_label_train, poi_neg_train = TrainDataValLoader(args.train_file, emb_weights,
                                                                           args.poi_file, args.batch_size)
    data_loader_val, poi_label_val, poi_neg_val = ValValueDataLoader(args.val_file, emb_weights,
                                                                     args.poi_file, args.batch_size)

    max_rec_v = -1
    max_epoch = -1

    for epoch in range(0, 200):
        model_name = 'epoch_' + str(epoch) + '.pt'

        model_f = '%s/%s' % (args.save_gcn_folder + '_' + str(args.alpha), model_name)
        if not os.path.exists(model_f):
            continue

        logging.info('Loading value nn from %s' % model_f)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        pbar = tqdm(data_loader_train)

        for batch_id, (batch_x, batch_y, batch_x_len, idx_list) in enumerate(pbar):
            batch_x = batch_x.to(device)
            last_output_emb, _ = model((batch_x, batch_x_len,
                                        poi_label_train.to(device), poi_neg_train.to(device)), True)
            emb_train[idx_list, :] = last_output_emb.detach()

        pbar = tqdm(data_loader_val)

        for batch_id, (batch_x, batch_y, batch_x_len, idx_list) in enumerate(pbar):
            batch_x = batch_x.to(device)
            last_output_emb, _ = model((batch_x, batch_x_len,
                                        poi_label_val.to(device), poi_neg_val.to(device)), True)
            rec[idx_list, :] = recall(last_output_emb, emb_train, batch_y, device, K).detach()

        rec_ave = rec.mean(axis=0)
        for recs in rec_ave:
            logging.info('%.4f' % recs)

            if recs > max_rec_v:
                max_rec_v = recs
                max_epoch = epoch

    logging.info(str(max_epoch))


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('Gcn_LSTM_eva_%.2f.log' % args.alpha)

    eval_model()
