import numpy as np
import torch
import random
from LSTM_Model import LSTM_Tra
from pars_args import args
from Data_Loader import GcnTrainValueDataLoader
from Trainer import Trainer
from logg import setup_logger


def train():
    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

    train_data_loader, emb_weights, poi_label, poi_neg = GcnTrainValueDataLoader(args.train_file, args.emb_file,
                                                                                 args.poi_file, args.batch_size)

    model = LSTM_Tra(args.n_layers, args.latent_dim, 0.5, device, emb_weights, batch_first=True)
    # model.load_state_dict(torch.load(r'E:\PyCharm_Project\OTRA_train_with_GAT\saved_models_0.5\epoch_19.pt'))

    trainer = Trainer(
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=None,
        poi_label=poi_label,
        poi_neg=poi_neg,
        n_epochs=args.n_epochs,
        alpha=args.alpha,
        lr=args.lr,
        save_epoch_int=args.save_epoch_int,
        model_folder=args.save_folder + '_' + str(args.alpha),
        device=device
    )

    trainer.train()


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('gcn_train_alpha_%.1f.log' % args.alpha)

    train()
