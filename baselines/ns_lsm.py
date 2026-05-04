"""
LSM (Latent Spectral Models) on NS ν=1e-5 — adapted to our 850/150/200 split.

Original: https://github.com/thuml/Latent-Spectral-Models  (Wu et al. 2023, ICML)
Source class: thuml/Neural-Solver-Library/models/LSM.py
Changes vs NSL defaults:
  - ntrain=850, nval=150, ntest=200 (was 1000/200)
  - data indices: train[:850], val[850:1000], test[-200:]
  - epochs=200 (was 500) — matches our Transolver / FactFormer / GNOT / ONO reruns
  - OneCycleLR (matches Transolver scaffold), not StepLR
  - Best-val checkpoint + per-sample test CSV output
  - Defaults = paper NS settings: n_hidden=64, n_heads=8, unified_pos=0, ref=8

Run from: ./third_party/NSL/
"""
import os, sys, csv, json
import scipy.io as scio
import numpy as np
import torch
from tqdm import tqdm
import argparse
from argparse import Namespace

parser = argparse.ArgumentParser('LSM NS — our split')
parser.add_argument('--lr',           type=float, default=5e-4)
parser.add_argument('--epochs',       type=int,   default=200)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--n-hidden',     type=int,   default=64)
parser.add_argument('--n-layers',     type=int,   default=8)
parser.add_argument('--n-heads',      type=int,   default=8)
parser.add_argument('--batch-size',   type=int,   default=20)
parser.add_argument('--gpu',          type=str,   default='0')
parser.add_argument('--max_grad_norm',type=float, default=None)
parser.add_argument('--downsample',   type=int,   default=1)
parser.add_argument('--unified_pos',  type=int,   default=0)
parser.add_argument('--ref',          type=int,   default=8)
parser.add_argument('--modes',        type=int,   default=12)
parser.add_argument('--act',          type=str,   default='gelu')
parser.add_argument('--save_name',    type=str,   default='ns_lsm_our_split')
parser.add_argument('--data_path',    type=str,   default='./data/NavierStokes_V1e-5_N1200_T20.mat')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# We're launched from ./third_party/NSL/ so models/ and utils/ are importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.LSM import Model as LSMModel
from utils.loss import L2Loss  # identical to Transolver's TestLoss.rel

torch.manual_seed(0)
np.random.seed(0)

ntrain, nval, ntest = 850, 150, 200
T_in, T, step = 10, 10, 1
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('plots',       exist_ok=True)


def eval_checkpoint(ckpt_path, model, loader, n_samples, device):
    """Autoregressive evaluation. Returns (mean_l2, per_sample_list)."""
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    total = 0.0
    per   = []
    loss_fn = L2Loss(size_average=False, reduction=False)  # per-sample
    with torch.no_grad():
        for x, fx, yy in loader:
            x, fx, yy = x.to(device), fx.to(device), yy.to(device)
            bsz = x.shape[0]
            pred = None
            for t in range(0, T, step):
                im   = model(x, fx=fx)
                pred = im if pred is None else torch.cat((pred, im), -1)
                fx   = torch.cat((fx[..., step:], im), dim=-1)
            per_batch = loss_fn(pred.reshape(bsz, -1), yy.reshape(bsz, -1))
            per.extend(per_batch.cpu().tolist())
            total += per_batch.sum().item()
    return total / n_samples, per


def main():
    r, h = args.downsample, int(((64-1)/args.downsample)+1)
    device = torch.device('cuda')

    u = scio.loadmat(args.data_path)['u']          # (N, 64, 64, T_total)
    print(f'Loaded: {u.shape}')

    def prep(s):
        a = torch.from_numpy(u[s, ::r, ::r, :T_in][:, :h, :h, :].reshape(-1, h*h, T_in))
        b = torch.from_numpy(u[s, ::r, ::r, T_in:T+T_in][:, :h, :h, :].reshape(-1, h*h, T))
        return a, b

    train_a, train_u = prep(slice(0, ntrain))
    val_a,   val_u   = prep(slice(ntrain, ntrain+nval))
    test_a,  test_u  = prep(slice(-ntest, None))

    xc, yc = np.meshgrid(np.linspace(0,1,h), np.linspace(0,1,h))
    pos = torch.tensor(np.c_[xc.ravel(), yc.ravel()], dtype=torch.float).unsqueeze(0)

    def make_loader(a, u, n, shuffle):
        gen = torch.Generator().manual_seed(0) if shuffle else None
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos.repeat(n,1,1), a, u),
            batch_size=args.batch_size, shuffle=shuffle, generator=gen)

    train_loader = make_loader(train_a, train_u, ntrain, True)
    val_loader   = make_loader(val_a,   val_u,   nval,   False)
    test_loader  = make_loader(test_a,  test_u,  ntest,  False)

    # Build the namespace LSM expects in its __init__
    model_args = Namespace(
        task='dynamic_autoregressive',
        geotype='structured_2D',
        shapelist=[h, h],
        unified_pos=args.unified_pos,
        ref=args.ref,
        fun_dim=T_in,
        space_dim=2,
        out_dim=step,           # 1 timestep per forward call (autoregressive)
        n_hidden=args.n_hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        act=args.act,
        time_input=False,
        modes=args.modes,
    )
    model = LSMModel(model_args).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    train_loss_fn = L2Loss(size_average=False)
    val_loss_fn   = L2Loss(size_average=False)

    best_val, best_epoch = float('inf'), -1

    losses_path = f'plots/{args.save_name}_losses.csv'
    losses_file = open(losses_path, 'w', newline='')
    losses_writer = csv.writer(losses_file)
    losses_writer.writerow(['epoch', 'train_l2', 'val_l2'])

    for ep in tqdm(range(args.epochs), desc='LSM NS'):
        model.train()
        train_full = 0.0
        for x, fx, yy in train_loader:
            x, fx, yy = x.to(device), fx.to(device), yy.to(device)
            bsz = x.shape[0]
            loss, pred = 0, None
            for t in range(0, T, step):
                y    = yy[..., t:t+step]
                im   = model(x, fx=fx)
                loss = loss + train_loss_fn(im.reshape(bsz,-1), y.reshape(bsz,-1))
                pred = im if pred is None else torch.cat((pred, im), -1)
                fx   = torch.cat((fx[..., step:], y), dim=-1)
            train_full += train_loss_fn(pred.reshape(bsz,-1), yy.reshape(bsz,-1)).item()
            optimizer.zero_grad(); loss.backward()
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); scheduler.step()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, fx, yy in val_loader:
                x, fx, yy = x.to(device), fx.to(device), yy.to(device)
                bsz = x.shape[0]; pred = None
                for t in range(0, T, step):
                    im   = model(x, fx=fx)
                    pred = im if pred is None else torch.cat((pred, im), -1)
                    fx   = torch.cat((fx[..., step:], im), dim=-1)
                val_total += val_loss_fn(pred.reshape(bsz,-1), yy.reshape(bsz,-1)).item()
        val_l2   = val_total / nval
        train_l2 = train_full / ntrain

        losses_writer.writerow([ep + 1, train_l2, val_l2]); losses_file.flush()

        if val_l2 < best_val:
            best_val, best_epoch = val_l2, ep+1
            torch.save(model.state_dict(), f'checkpoints/{args.save_name}_best.pt')

        if (ep+1) % 20 == 0 or ep == args.epochs-1:
            print(f'Ep {ep+1:3d}  train={train_l2:.4f}  val={val_l2:.4f}  best={best_val:.4f}@{best_epoch}')

    losses_file.close()
    torch.save(model.state_dict(), f'checkpoints/{args.save_name}_final.pt')
    print(f'\nBest val: epoch {best_epoch}, val L2={best_val:.4f}')

    best_test_l2,  per_best  = eval_checkpoint(f'checkpoints/{args.save_name}_best.pt',  model, test_loader, ntest, device)
    final_test_l2, per_final = eval_checkpoint(f'checkpoints/{args.save_name}_final.pt', model, test_loader, ntest, device)

    print(f'\nBest-val  test L2 = {best_test_l2:.4f}  (epoch {best_epoch})')
    print(f'Final-ep  test L2 = {final_test_l2:.4f}  (epoch {args.epochs})')
    print(f'Overfit gap        = {final_test_l2 - best_test_l2:+.4f}')

    tag = args.save_name
    with open(f'plots/{tag}_test_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_idx', 'test_l2_best', 'test_l2_final'])
        for i, (b, fin) in enumerate(zip(per_best, per_final)):
            w.writerow([i, b, fin])

    cfg = {**vars(args), 'ntrain': ntrain, 'nval': nval, 'ntest': ntest,
           'T_in': T_in, 'T_out': T, 'step': step,
           'n_params': n_params, 'best_epoch': best_epoch,
           'best_test_l2': best_test_l2, 'final_test_l2': final_test_l2}
    with open(f'plots/{tag}_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'Results saved to plots/{tag}_test_results.csv')


if __name__ == '__main__':
    main()
