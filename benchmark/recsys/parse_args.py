import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Image inpaiting.')

    parser.add_argument("--n_core", type=int, default=20, help="")

    parser.add_argument("--hidden_size", type=int, default=128, help="")

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu_idx', type=str, default='7')
    parser.add_argument("--train_ratio", type=float, default=0.8, help="")
    parser.add_argument("--debug", default=False, help="")
    parser.add_argument("--epochs", type=int, default=40, help="")
    parser.add_argument("--batch_size", type=int, default=1024, help="")
    parser.add_argument("--lr", type=float, default=1e-3, help="")
    parser.add_argument("--weight_decay", type=float, default=0, help="")

    parser.add_argument("--emb_dim", type=int, default=300, help="")
    parser.add_argument("--repr_dim", type=int, default=64, help="")

    args = parser.parse_args()

    return args
