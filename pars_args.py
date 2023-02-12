import argparse
import os


parser = argparse.ArgumentParser()

# ===================== gpu id ===================== #
parser.add_argument('--gpu',            type=int,   default=0)

# =================== random seed ================== #
parser.add_argument('--seed',           type=int,   default=1234)

# ==================== dataset ===================== #
parser.add_argument('--train_file',
                    default='data/bj_train_set.npz')
parser.add_argument('--val_file',
                    default='data/bj_val_set.npz')
parser.add_argument('--test_file',
                    default='data/bj_test_set.npz')
parser.add_argument('--poi_file',
                    default='data/bj_transh_poi_10.npz')
parser.add_argument('--nodes',          type=int,   default=28342,
                    help='Newyork=95581, Beijing=28342')

# ===================== model ====================== #
parser.add_argument('--latent_dim',     type=int,   default=128)
parser.add_argument('--num_heads',      type=int,   default=8)
parser.add_argument('--lstm_layers',    type=int,   default=4)
parser.add_argument('--n_epochs',       type=int,   default=300)
parser.add_argument('--batch_size',     type=int,   default=256)
parser.add_argument('--lr',             type=float, default=5e-4,
                    help='5e-4 for Beijing, 1e-3 for Newyork')  # 2e-3
parser.add_argument('--save_epoch_int', type=int,   default=1)
parser.add_argument('--save_folder',                default='saved_models')

args = parser.parse_args()

# setup device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
