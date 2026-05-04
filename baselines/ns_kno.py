"""
KNO2d (Koopman Neural Operator) on NS ν=1e-5 — adapted to our 850/150/200 split.

Original: https://github.com/Koopman-Laboratory/KoopmanLab  (demo_ns.py)
Changes:
  - ntrain=850, nval=150, ntest=200 (was 1000/200)
  - data: NavierStokes_V1e-5_N1200_T20.mat (our .mat format, not h5py)
  - epochs=500 (matches our other baselines)
  - KNO is NOT autoregressive: encoder takes full T_in, decoder outputs full T_out
  - encoder_conv2d(t_len=T_in), decoder_conv2d(t_len=T_out)
  - Loss = relative L2 (same as all other baselines)
  - KNO forward returns (pred, reconstruction); reconstruction loss added as reg
  - Best-val ckpt + per-sample test CSV output

Run from: ./third_party/KoopmanLab/
"""
import os, sys, csv, json
import scipy.io as scio
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser('KNO2d NS — our split')
parser.add_argument('--lr',           type=float, default=5e-3)
parser.add_argument('--epochs',       type=int,   default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--op_size',      type=int,   default=32,  help='Koopman operator size')
parser.add_argument('--modes',        type=int,   default=16,  help='Fourier modes')
parser.add_argument('--decompose',    type=int,   default=6,   help='Koopman decompose depth')
parser.add_argument('--batch-size',   type=int,   default=10)
parser.add_argument('--gpu',          type=str,   default='0')
parser.add_argument('--max_grad_norm',type=float, default=None)
parser.add_argument('--step_size',    type=int,   default=100)
parser.add_argument('--gamma',        type=float, default=0.5)
parser.add_argument('--rec_weight',   type=float, default=1.0, help='reconstruction loss weight')
parser.add_argument('--save_name',    type=str,   default='ns_kno_our_split')
parser.add_argument('--data_path',    type=str,   default='./data/NavierStokes_V1e-5_N1200_T20.mat')
parser.add_argument('--resume',       action='store_true', help='Resume from checkpoints/{save_name}_checkpoint.pt')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Import kno.py directly — the package __init__ pulls in koopmanViT which uses
# numpy.lib.arraypad (removed in numpy ≥1.25). Direct import avoids that chain.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "kno_module",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "koopmanlab", "models", "kno.py"))
_kno = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_kno)
KNO2d = _kno.KNO2d
encoder_conv2d = _kno.encoder_conv2d
decoder_conv2d = _kno.decoder_conv2d

torch.manual_seed(0)
np.random.seed(0)

ntrain, nval, ntest = 850, 150, 200
T_in, T = 10, 10
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('plots',       exist_ok=True)

H = W = 64  # spatial resolution


def rel_l2_batch(pred, target):
    """Per-sample relative L2. pred/target: (B, H, W) or (B, *)."""
    B = pred.shape[0]
    diff = (pred - target).reshape(B, -1).norm(dim=1)
    norm = target.reshape(B, -1).norm(dim=1)
    return diff / (norm + 1e-8)  # (B,)


def eval_checkpoint(ckpt_path, model, loader, n_samples, device):
    """Single-shot (non-autoregressive) evaluation."""
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    total, per = 0.0, []
    with torch.no_grad():
        for inp, target in loader:
            inp, target = inp.to(device), target.to(device)
            bsz = inp.shape[0]
            pred, _ = model(inp)           # (B, H, W, T_out)
            per_batch = rel_l2_batch(pred.reshape(bsz, -1), target.reshape(bsz, -1))
            per.extend(per_batch.cpu().tolist())
            total += per_batch.sum().item()
    return total / n_samples, per


def main():
    device = torch.device('cuda')

    u = scio.loadmat(args.data_path)['u']   # (1200, 64, 64, 20)
    u = torch.from_numpy(u).float()         # float32

    def prep(s):
        a = u[s, :, :, :T_in]              # (N, H, W, T_in)
        b = u[s, :, :, T_in:T_in+T]        # (N, H, W, T)
        return a, b

    train_a, train_u = prep(slice(0, ntrain))
    val_a,   val_u   = prep(slice(ntrain, ntrain+nval))
    test_a,  test_u  = prep(slice(-ntest, None))

    def make_loader(a, u_t, shuffle):
        gen = torch.Generator().manual_seed(0) if shuffle else None
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(a, u_t),
            batch_size=args.batch_size, shuffle=shuffle, generator=gen)

    train_loader = make_loader(train_a, train_u, True)
    val_loader   = make_loader(val_a,   val_u,   False)
    test_loader  = make_loader(test_a,  test_u,  False)

    # KNO2d: encoder takes T_in channels, decoder outputs T_out channels (single-shot)
    enc = encoder_conv2d(t_len=T_in, op_size=args.op_size)
    dec = decoder_conv2d(t_len=T,    op_size=args.op_size)
    model = KNO2d(enc, dec, op_size=args.op_size,
                  modes_x=args.modes, modes_y=args.modes,
                  decompose=args.decompose, linear_type=True).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[KNO2d] Parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    best_val, best_epoch = float('inf'), -1
    start_epoch = 0
    ckpt_path = f'checkpoints/{args.save_name}_checkpoint.pt'
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt['best_val']
        best_epoch  = ckpt['best_epoch']
        print(f'[KNO2d] Resumed from epoch {ckpt["epoch"]}, best_val={best_val:.4f}')

    losses_path = f'plots/{args.save_name}_losses.csv'
    csv_mode = 'a' if (args.resume and os.path.exists(ckpt_path)) else 'w'
    losses_file = open(losses_path, csv_mode, newline='')
    losses_writer = csv.writer(losses_file)
    if csv_mode == 'w':
        losses_writer.writerow(['epoch', 'train_l2', 'val_l2'])

    for ep in tqdm(range(start_epoch, args.epochs), desc='KNO2d NS'):
        model.train()
        train_full = 0.0
        for inp, target in train_loader:
            inp, target = inp.to(device), target.to(device)
            bsz = inp.shape[0]
            # Single-shot: inp (B,H,W,T_in) → pred (B,H,W,T_out)
            pred, rec = model(inp)
            step_loss = rel_l2_batch(pred.reshape(bsz,-1), target.reshape(bsz,-1)).mean()
            rec_loss  = rel_l2_batch(rec.reshape(bsz,-1),  inp.reshape(bsz,-1)).mean()
            loss = step_loss + args.rec_weight * rec_loss
            train_full += rel_l2_batch(pred.reshape(bsz,-1), target.reshape(bsz,-1)).sum().item()
            optimizer.zero_grad(); loss.backward()
            if args.max_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); scheduler.step()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for inp, target in val_loader:
                inp, target = inp.to(device), target.to(device)
                bsz = inp.shape[0]
                pred, _ = model(inp)
                val_total += rel_l2_batch(pred.reshape(bsz,-1), target.reshape(bsz,-1)).sum().item()
        val_l2   = val_total / nval
        train_l2 = train_full / ntrain

        losses_writer.writerow([ep+1, train_l2, val_l2]); losses_file.flush()

        if val_l2 < best_val:
            best_val, best_epoch = val_l2, ep+1
            torch.save(model.state_dict(), f'checkpoints/{args.save_name}_best.pt')

        torch.save({
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'epoch':      ep,
            'best_val':   best_val,
            'best_epoch': best_epoch,
        }, f'checkpoints/{args.save_name}_checkpoint.pt')

        if (ep+1) % 20 == 0 or ep == args.epochs-1:
            print(f'[KNO2d] Ep {ep+1:3d}  train={train_l2:.4f}  val={val_l2:.4f}  best={best_val:.4f}@{best_epoch}')

    losses_file.close()
    torch.save(model.state_dict(), f'checkpoints/{args.save_name}_final.pt')

    best_test_l2,  per_best  = eval_checkpoint(f'checkpoints/{args.save_name}_best.pt',  model, test_loader, ntest, device)
    final_test_l2, per_final = eval_checkpoint(f'checkpoints/{args.save_name}_final.pt', model, test_loader, ntest, device)

    print(f'\n[KNO2d] Best-val  test L2 = {best_test_l2:.4f}  (epoch {best_epoch})')
    print(f'[KNO2d] Final-ep  test L2 = {final_test_l2:.4f}  (epoch {args.epochs})')
    print(f'[KNO2d] Overfit gap        = {final_test_l2 - best_test_l2:+.4f}')

    tag = args.save_name
    with open(f'plots/{tag}_test_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_idx', 'test_l2_best', 'test_l2_final'])
        for i, (b, fin) in enumerate(zip(per_best, per_final)):
            w.writerow([i, b, fin])

    cfg = {**vars(args), 'model_name': 'KNO2d', 'ntrain': ntrain, 'nval': nval, 'ntest': ntest,
           'T_in': T_in, 'T_out': T,
           'n_params': n_params, 'best_epoch': best_epoch,
           'best_test_l2': best_test_l2, 'final_test_l2': final_test_l2}
    with open(f'plots/{tag}_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'[KNO2d] Results saved to plots/{tag}_test_results.csv')


if __name__ == '__main__':
    main()
