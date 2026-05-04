"""
GNOT (CGPTNO) on NS ν=1e-5 — adapted to our 850/150/200 split with FNO mat data.

Original: https://github.com/HaoZhongkai/GNOT  (gnot_exp.sh, NS settings)
Changes:
  - Data: NavierStokes_V1e-5_N1200_T20.mat (64×64), not original NS data
  - ntrain=850, nval=150, ntest=200
  - epochs=200
  - Best-val checkpoint + per-sample test CSV output
  - Model: CGPTNO n_hidden=128, n_layers=3, n_head=1, attn_type='linear'
  - Loss: relative L2 = ||pred-target||_2 / ||target||_2, same as all other baselines
  - No output normalisation — raw vorticity values throughout (keeps metrics comparable)

Run from: ./third_party/gnot/
"""
import os, sys, csv, json
import scipy.io as scio
import numpy as np
import torch
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser('GNOT CGPTNO NS — our split')
parser.add_argument('--lr',            type=float, default=1e-3)
parser.add_argument('--epochs',        type=int,   default=200)
parser.add_argument('--weight_decay',  type=float, default=5e-5)
parser.add_argument('--n_hidden',      type=int,   default=128)
parser.add_argument('--n_layers',      type=int,   default=3)
parser.add_argument('--n_heads',       type=int,   default=1)
parser.add_argument('--batch_size',    type=int,   default=4)
parser.add_argument('--attn_type',     type=str,   default='linear')
parser.add_argument('--max_grad_norm', type=float, default=1000.0)
parser.add_argument('--gpu',           type=str,   default='0')
parser.add_argument('--save_name',     type=str,   default='ns_gnot_our_split')
parser.add_argument('--data_path',     type=str,   default='./data/NavierStokes_V1e-5_N1200_T20.mat')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dgl
from models.cgpt import CGPTNO
from data_utils import FNODataset, MIODataLoader

ntrain, nval, ntest = 850, 150, 200
T_in, T_out = 10, 10
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('plots',       exist_ok=True)


# ---------------------------------------------------------------------------
# Loss helpers — work directly on DGL batched graphs
# ---------------------------------------------------------------------------

def dgl_rel_l2_mean(pred, target, g):
    """
    Batch-mean relative L2.  pred / target: (total_nodes, T_out).
    Splits the batched graph into individual graphs to get per-sample norms.
    Returns a scalar (differentiable through pred).
    """
    node_counts = g.batch_num_nodes()   # (B,) int tensor
    losses = []
    offset = 0
    for n in node_counts.tolist():
        p = pred[offset:offset+n]       # (n_nodes, T_out)
        t = target[offset:offset+n]
        losses.append((p - t).norm() / (t.norm() + 1e-8))
        offset += n
    return torch.stack(losses).mean()


def dgl_rel_l2_sum(pred, target, g):
    """Same as above but returns sum (for accumulating epoch totals)."""
    node_counts = g.batch_num_nodes()
    losses = []
    offset = 0
    for n in node_counts.tolist():
        p = pred[offset:offset+n]
        t = target[offset:offset+n]
        losses.append((p - t).norm() / (t.norm() + 1e-8))
        offset += n
    return torch.stack(losses).sum()


def per_sample_rel_l2(pred, target, g):
    """Returns list of per-sample relative L2 (numpy floats)."""
    node_counts = g.batch_num_nodes()
    per = []
    offset = 0
    for n in node_counts.tolist():
        p = pred[offset:offset+n].reshape(-1)
        t = target[offset:offset+n].reshape(-1)
        per.append(float((p - t).norm() / (t.norm() + 1e-8)))
        offset += n
    return per


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda')

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    u = scio.loadmat(args.data_path)['u']       # (N, 64, 64, T_total)
    N_total = u.shape[0]
    print(f'Loaded: {u.shape}')
    assert N_total >= ntrain + nval + ntest, \
        f'Need {ntrain+nval+ntest} samples, got {N_total}'

    # 2-D coordinate grid in [0, 1]²
    grid_1d = np.linspace(0, 1, 64, dtype=np.float32)
    gx, gy  = np.meshgrid(grid_1d, grid_1d, indexing='ij')   # (64, 64)
    coords  = np.stack([gx, gy], axis=-1).reshape(4096, 2)   # (4096, 2)

    def make_XY(idx_slice):
        u_sub = u[idx_slice]                                        # (n, 64, 64, T_total)
        n     = u_sub.shape[0]
        u_in  = u_sub[:, :, :, :T_in].reshape(n, 4096, T_in).astype(np.float32)
        u_out = u_sub[:, :, :, T_in:T_in+T_out].reshape(n, 4096, T_out).astype(np.float32)
        c_rep = np.tile(coords[None], (n, 1, 1))                   # (n, 4096, 2)
        X = np.concatenate([c_rep, u_in], axis=-1)                 # (n, 4096, 12)
        return X, u_out

    X_train, Y_train = make_XY(slice(0, ntrain))
    X_val,   Y_val   = make_XY(slice(ntrain, ntrain + nval))
    X_test,  Y_test  = make_XY(slice(-ntest, None))

    print(f'X_train: {X_train.shape}  Y_train: {Y_train.shape}')

    # ------------------------------------------------------------------
    # Datasets — no output normalisation (raw values, comparable to FNO/SUNet)
    # ------------------------------------------------------------------
    train_dataset = FNODataset(X_train, Y_train, normalize_y=False)
    val_dataset   = FNODataset(X_val,   Y_val,   normalize_y=False)
    test_dataset  = FNODataset(X_test,  Y_test,  normalize_y=False)

    train_loader = MIODataLoader(train_dataset, batch_size=args.batch_size,
                                 shuffle=True,  drop_last=False)
    val_loader   = MIODataLoader(val_dataset,   batch_size=args.batch_size,
                                 shuffle=False, drop_last=False)
    test_loader  = MIODataLoader(test_dataset,  batch_size=args.batch_size,
                                 shuffle=False, drop_last=False)

    # ------------------------------------------------------------------
    # Model — matches gnot_exp.sh NS settings
    # ------------------------------------------------------------------
    input_dim  = T_in + 2   # 12: (x,y) + 10 vorticity snapshots
    output_dim = T_out       # 10

    model = CGPTNO(
        trunk_size   = input_dim + 1,
        branch_sizes = None,
        output_size  = output_dim,
        n_layers     = args.n_layers,
        n_hidden     = args.n_hidden,
        n_head       = args.n_heads,
        attn_type    = args.attn_type,
        ffn_dropout  = 0.0,
        attn_dropout = 0.0,
        mlp_layers   = 3,
        act          = 'gelu',
        horiz_fourier_dim = 0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader))

    best_val, best_epoch = float('inf'), -1

    losses_path = f'plots/{args.save_name}_losses.csv'
    losses_file = open(losses_path, 'w', newline='')
    losses_writer = csv.writer(losses_file)
    losses_writer.writerow(['epoch', 'train_l2', 'val_l2'])

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for ep in tqdm(range(args.epochs), desc='GNOT NS'):
        model.train()
        train_sum = 0.0
        for g, u_p, g_u in train_loader:
            g   = g.to(device)
            u_p = u_p.to(device)
            g_u = g_u.to(device) if g_u is not None else None

            out  = model(g, u_p, g_u)                    # (total_nodes, T_out)
            loss = dgl_rel_l2_mean(out, g.ndata['y'], g)
            bsz  = g.batch_num_nodes().shape[0]
            train_sum += loss.item() * bsz

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for g, u_p, g_u in val_loader:
                g   = g.to(device)
                u_p = u_p.to(device)
                g_u = g_u.to(device) if g_u is not None else None
                out = model(g, u_p, g_u)
                val_total += dgl_rel_l2_sum(out, g.ndata['y'], g).item()

        val_l2 = val_total / nval
        train_l2 = train_sum / ntrain

        losses_writer.writerow([ep + 1, train_l2, val_l2])
        losses_file.flush()

        if val_l2 < best_val:
            best_val, best_epoch = val_l2, ep + 1
            torch.save(model.state_dict(), f'checkpoints/{args.save_name}_best.pt')

        if (ep + 1) % 20 == 0 or ep == args.epochs - 1:
            print(f'Ep {ep+1:3d}  train={train_l2:.4f}  '
                  f'val={val_l2:.4f}  best={best_val:.4f}@{best_epoch}')

    losses_file.close()
    torch.save(model.state_dict(), f'checkpoints/{args.save_name}_final.pt')
    print(f'\nBest val: epoch {best_epoch}, val L2={best_val:.4f}')

    # ------------------------------------------------------------------
    # Test evaluation — per-sample relative L2
    # ------------------------------------------------------------------
    def eval_checkpoint(ckpt_path, tag):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        total, per = 0.0, []
        with torch.no_grad():
            for g, u_p, g_u in test_loader:
                g   = g.to(device)
                u_p = u_p.to(device)
                g_u = g_u.to(device) if g_u is not None else None
                out = model(g, u_p, g_u)
                batch_per = per_sample_rel_l2(out, g.ndata['y'], g)
                per.extend(batch_per)
                total += sum(batch_per)
        mean_l2 = total / ntest
        print(f'{tag:12s}  test L2 = {mean_l2:.4f}')
        return mean_l2, per

    best_test_l2,  per_best  = eval_checkpoint(
        f'checkpoints/{args.save_name}_best.pt',  'best-val')
    final_test_l2, per_final = eval_checkpoint(
        f'checkpoints/{args.save_name}_final.pt', 'final-ep')

    print(f'\nBest-val  test L2 = {best_test_l2:.4f}  (epoch {best_epoch})')
    print(f'Final-ep  test L2 = {final_test_l2:.4f}  (epoch {args.epochs})')
    print(f'Overfit gap        = {final_test_l2 - best_test_l2:+.4f}')
    print(f'(Transolver paper GNOT NS: 0.1543)')

    tag = args.save_name
    with open(f'plots/{tag}_test_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_idx', 'test_l2_best', 'test_l2_final'])
        for i, (b, fin) in enumerate(zip(per_best, per_final)):
            w.writerow([i, b, fin])

    cfg = {**vars(args), 'ntrain': ntrain, 'nval': nval, 'ntest': ntest,
           'n_params': n_params, 'best_epoch': best_epoch,
           'best_test_l2': best_test_l2, 'final_test_l2': final_test_l2}
    with open(f'plots/{tag}_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'Results saved to plots/{tag}_test_results.csv')


if __name__ == '__main__':
    main()
