from GRLSTM_Model import ResLSTM
from pars_args import args
from Data_Loader import TrainValueDataLoader
from Trainer import Trainer
from logg import setup_logger

import numpy as np
import torch
import random


def train():
    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

    train_data_loader = TrainValueDataLoader(args.train_file, args.poi_file, args.batch_size)

    model = ResLSTM(args.nodes, args.latent_dim, device, args.poi_file, batch_first=True)

    # model.load_state_dict(torch.load('saved_models_tmp/epoch_146.pt'))

    trainer = Trainer(
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=None,
        n_epochs=args.n_epochs,
        lr=args.lr,
        save_epoch_int=args.save_epoch_int,
        model_folder=args.save_folder,
        device=device
    )

    trainer.train()


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('GRLSTM_train.log')
    train()
