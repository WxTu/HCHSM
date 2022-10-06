from utils import *
from train import Train

if __name__ == '__main__':
    import warnings
    import argparse

    parser = argparse.ArgumentParser(description='model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='citeseer')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sample_size', type=int, default=3327)
    parser.add_argument('--small_ratio', type=int, default=1)
    parser.add_argument('--n_clusters', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--nb_epochs', type=int, default=120)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--cnt_wait', type=int, default=0)
    parser.add_argument('--best_t', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--l2_coef', type=float, default=0.0)
    parser.add_argument('--drop_rate', type=float, default=0.3)
    parser.add_argument('--best', type=float, default=1e9)
    parser.add_argument('--hid_units', type=int, default=512)
    parser.add_argument('--sparse', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    torch.cuda.set_device(0)

    print("training beginning...")
    acc_list = []

    setup_seed(args.seed)
    acc = Train(dataset=args.name, nb_epochs=args.nb_epochs, patience=args.patience, lr=args.lr,
                l2_coef=args.l2_coef, hid_units=args.hid_units, sample_size=args.sample_size,
                batch_size=args.batch_size, small_ratio_=args.small_ratio, cnt_wait=args.cnt_wait,
                best=args.best, best_t=args.best_t, gamma=args.gamma, sigma=args.sigma, drop_rate=args.drop_rate,
                cuda=args.cuda, sparse=args.sparse, verbose=args.verbose)



