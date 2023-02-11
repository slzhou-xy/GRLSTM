import numpy as np
import torch
import random

import logging
from logg import setup_logger
from pars_args import args
from GRLSTM_Model import GRLSTM
from Data_Loader import load_traindata
from Data_Loader import load_testdata
from Data_Loader import TestValueDataLoader
from Data_Loader import TrainDataValLoader


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

    train_x, train_y = load_traindata(args.train_file)
    val_x, val_y = load_testdata(args.test_file)

    emb_train = np.zeros((len(train_x), args.latent_dim))

    model = GRLSTM(args, device, batch_first=True).to(device)

    epoch = open('GRLSTM_eva.log').read().splitlines()[-1][-3:]

    model_name = 'epoch_' + epoch + '.pt'

    model_f = '%s/%s' % (args.save_folder, model_name)

    logging.info('Loading value nn from %s' % model_f)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()

    data_loader_val = TestValueDataLoader(args.test_file, args.batch_size)

    data_loader_train = TrainDataValLoader(args.train_file, args.batch_size)

    for batch_id, (batch_x, batch_y, batch_x_len, idx_list) in enumerate(data_loader_train):
        batch_x = batch_x.to(device)
        last_output_emb, _, _ = model((batch_x, batch_x_len, None, None), True)
        emb_train[idx_list, :] = last_output_emb.cpu().detach().numpy()

    K = [1, 5, 10, 20, 50]

    rec = np.zeros((val_x.shape[0], len(K)))
    for batch_id, (batch_x, batch_y, batch_x_len, idx_list) in enumerate(data_loader_val):
        batch_x = batch_x.to(device)
        last_output_emb, _, _ = model((batch_x, batch_x_len, None, None), True)
        rec[idx_list, :] = recall(
            last_output_emb.cpu().detach().numpy(), emb_train, batch_y, K)

    rec_ave = rec.mean(axis=0)
    for rec in rec_ave:
        logging.info('%.4f' % rec)

    print(rec_ave)


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('GRLSTM_test.log')
    eval_model()
