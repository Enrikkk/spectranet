"""
CNO2d (Convolutional Neural Operator, Raonic et al., NeurIPS 2023) on NS ν=1e-5 —
adapted to our 850/150/200 split, sliding-window autoregressive rollout.

Model source: ./third_party/CNO/CNO2d_simplified/CNO2d.py
  (pure PyTorch, no StyleGAN3 / Lightning / CUDA-kernel deps)

Protocol (gold-standard, identical to ns_sunet_ar2d.py / ns_oformer.py):
  - 850 train / 150 val / 200 test, T_in=10, T_out=10, step=1
  - Data: NavierStokes_V1e-5_N1200_T20.mat (.mat scipy format, 64×64)
  - Optimizer: AdamW, weight_decay=1e-5
  - Scheduler: OneCycleLR (max_lr=lr, stepped per batch)
  - Loss: relative L2 per-sample (LpLoss(size_average=False) summed across batch)
  - AR rollout: teacher forcing at train, free rollout at val/test
  - No normalizer (matches NSL-AR convention)

Run from: ./third_party/CNO/   (so importlib path resolves)
"""
import os, sys, csv, json
import argparse
import importlib.util as _ilu
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
from tqdm import tqdm

parser = argparse.ArgumentParser('CNO2d NS — our split')
parser.add_argument('--lr',                type=float, default=1e-3)
parser.add_argument('--epochs',            type=int,   default=500)
parser.add_argument('--weight_decay',      type=float, default=1e-5)
parser.add_argument('--batch_size',        type=int,   default=10)
parser.add_argument('--N_layers',          type=int,   default=3)
parser.add_argument('--N_res',             type=int,   default=4)
parser.add_argument('--N_res_neck',        type=int,   default=4)
parser.add_argument('--channel_multiplier',type=int,   default=32)
parser.add_argument('--use_bn',            type=int,   default=1, help='1=use batchnorm, 0=skip')
parser.add_argument('--scheduler',         type=str,   default='oclr', choices=['oclr','step'])
parser.add_argument('--seed',              type=int,   default=0)
parser.add_argument('--gpu',               type=str,   default='0')
parser.add_argument('--save_name',         type=str,   default='ns_CNO_our_split')
parser.add_argument('--data_path',         type=str,
                    default='./data/NavierStokes_V1e-5_N1200_T20.mat')
parser.add_argument('--cno_path',          type=str,
                    default='./third_party/CNO/CNO2d_simplified/CNO2d.py')
parser.add_argument('--resume',            action='store_true',
                    help='Resume from checkpoints/{save_name}_checkpoint.pt')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ----- import CNO2d via importlib (avoids polluting site-packages) -----
_spec = _ilu.spec_from_file_location('cno2d_module', args.cno_path)
_cno  = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_cno)
CNO2d = _cno.CNO2d

ntrain, nval, ntest = 850, 150, 200
T_in, T_out, step   = 10, 10, 1
H = W = 64

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('plots',       exist_ok=True)


def rel_l2_batch(pred, target):
    """Per-sample relative L2. pred/target: (B, *)."""
    B = pred.shape[0]
    diff = (pred - target).reshape(B, -1).norm(dim=1)
    norm = target.reshape(B, -1).norm(dim=1)
    return diff / (norm + 1e-8)


def free_rollout(model, x_window, T):
    """x_window: (B, H, W, T_in) -> pred (B, H, W, T) via free AR rollout."""
    cur = x_window
    pred_seq = []
    for _ in range(T):
        inp = cur.permute(0, 3, 1, 2).contiguous()        # (B, T_in, H, W)
        out = model(inp)                                   # (B, 1, H, W)
        out = out.permute(0, 2, 3, 1).contiguous()         # (B, H, W, 1)
        pred_seq.append(out)
        cur = torch.cat([cur[..., step:], out], dim=-1)
    return torch.cat(pred_seq, dim=-1)                     # (B, H, W, T)


def eval_checkpoint(ckpt_path, model, loader, n_samples, device):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'] if isinstance(state, dict) and 'model' in state else state)
    model.eval()
    total, per = 0.0, []
    with torch.no_grad():
        for x_window, y_full in loader:
            x_window, y_full = x_window.to(device), y_full.to(device)
            bsz = x_window.shape[0]
            pred = free_rollout(model, x_window, T_out)
            per_batch = rel_l2_batch(pred.reshape(bsz, -1), y_full.reshape(bsz, -1))
            per.extend(per_batch.cpu().tolist())
            total += per_batch.sum().item()
    return total / n_samples, per


def main():
    device = torch.device('cuda')

    u = scio.loadmat(args.data_path)['u']           # (1200, 64, 64, 20) numpy
    u = torch.from_numpy(u).float()

    def prep(s):
        a = u[s, :, :, :T_in]                       # (N, H, W, T_in)
        b = u[s, :, :, T_in:T_in+T_out]             # (N, H, W, T_out)
        return a, b

    train_a, train_u = prep(slice(0, ntrain))
    val_a,   val_u   = prep(slice(ntrain, ntrain+nval))
    test_a,  test_u  = prep(slice(-ntest, None))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed))
    val_loader   = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_a, val_u),
        batch_size=20, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u),
        batch_size=1, shuffle=False)

    model = CNO2d(
        in_dim=T_in,
        out_dim=1,
        size=H,
        N_layers=args.N_layers,
        N_res=args.N_res,
        N_res_neck=args.N_res_neck,
        channel_multiplier=args.channel_multiplier,
        use_bn=bool(args.use_bn),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[CNO] Parameters (sum.numel): {n_params:,}  trainable: {n_train_params:,}')
    print(f'[CNO] N_layers={args.N_layers}  channel_mult={args.channel_multiplier}  '
          f'N_res={args.N_res}  N_res_neck={args.N_res_neck}  use_bn={bool(args.use_bn)}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    if args.scheduler == 'oclr':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr,
            epochs=args.epochs, steps_per_epoch=len(train_loader))
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    best_val, best_epoch = float('inf'), -1
    start_epoch = 0
    ckpt_path = f'checkpoints/{args.save_name}_checkpoint.pt'
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val   = ckpt['best_val']
        best_epoch = ckpt['best_epoch']
        print(f'[CNO] Resumed from ep {ckpt["epoch"]}, best_val={best_val:.4f}')

    losses_path = f'plots/{args.save_name}_losses.csv'
    csv_mode = 'a' if (args.resume and os.path.exists(ckpt_path)) else 'w'
    losses_file = open(losses_path, csv_mode, newline='')
    losses_writer = csv.writer(losses_file)
    if csv_mode == 'w':
        losses_writer.writerow(['epoch', 'train_l2', 'val_l2'])

    for ep in tqdm(range(start_epoch, args.epochs), desc='CNO NS'):
        model.train()
        train_total = 0.0
        for x_window, y_full in train_loader:
            x_window, y_full = x_window.to(device), y_full.to(device)
            bsz = x_window.shape[0]
            cur = x_window
            loss = torch.tensor(0.0, device=device)
            pred_seq = []
            for t in range(T_out):
                y_t = y_full[..., t:t+step]                       # (B, H, W, 1)
                inp = cur.permute(0, 3, 1, 2).contiguous()        # (B, T_in, H, W)
                out = model(inp).permute(0, 2, 3, 1).contiguous() # (B, H, W, 1)
                loss = loss + rel_l2_batch(out.reshape(bsz, -1),
                                           y_t.reshape(bsz, -1)).mean()
                pred_seq.append(out)
                # teacher forcing: feed ground-truth next frame, not pred
                cur = torch.cat([cur[..., step:], y_t], dim=-1)
            pred = torch.cat(pred_seq, dim=-1)
            train_total += rel_l2_batch(pred.reshape(bsz, -1),
                                        y_full.reshape(bsz, -1)).sum().item()
            optimizer.zero_grad(); loss.backward()
            optimizer.step()
            if args.scheduler == 'oclr':
                scheduler.step()
        if args.scheduler == 'step':
            scheduler.step()

        # validation: free rollout
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x_window, y_full in val_loader:
                x_window, y_full = x_window.to(device), y_full.to(device)
                bsz  = x_window.shape[0]
                pred = free_rollout(model, x_window, T_out)
                val_total += rel_l2_batch(pred.reshape(bsz, -1),
                                          y_full.reshape(bsz, -1)).sum().item()
        train_l2 = train_total / ntrain
        val_l2   = val_total   / nval

        losses_writer.writerow([ep+1, train_l2, val_l2]); losses_file.flush()

        if val_l2 < best_val:
            best_val, best_epoch = val_l2, ep+1
            torch.save({'model': model.state_dict()},
                       f'checkpoints/{args.save_name}_best.pt')

        torch.save({
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'epoch':      ep,
            'best_val':   best_val,
            'best_epoch': best_epoch,
        }, ckpt_path)

        if (ep+1) % 20 == 0 or ep == args.epochs-1:
            print(f'[CNO] Ep {ep+1:3d}  train={train_l2:.4f}  val={val_l2:.4f}  '
                  f'best={best_val:.4f}@{best_epoch}')

    losses_file.close()
    torch.save({'model': model.state_dict()},
               f'checkpoints/{args.save_name}_final.pt')

    best_test_l2,  per_best  = eval_checkpoint(
        f'checkpoints/{args.save_name}_best.pt',  model, test_loader, ntest, device)
    final_test_l2, per_final = eval_checkpoint(
        f'checkpoints/{args.save_name}_final.pt', model, test_loader, ntest, device)

    print(f'\n[CNO] Best-val  test L2 = {best_test_l2:.4f}  (epoch {best_epoch})')
    print(f'[CNO] Final-ep  test L2 = {final_test_l2:.4f}  (epoch {args.epochs})')

    tag = args.save_name
    with open(f'plots/{tag}_test_results.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['sample_idx', 'test_l2_best', 'test_l2_final'])
        for i, (b, fin) in enumerate(zip(per_best, per_final)):
            w.writerow([i, b, fin])

    cfg = {**vars(args), 'model_name': 'CNO',
           'ntrain': ntrain, 'nval': nval, 'ntest': ntest,
           'T_in': T_in, 'T_out': T_out, 'step': step,
           'n_params': n_params, 'best_val_epoch': best_epoch,
           'best_test_l2': best_test_l2, 'final_test_l2': final_test_l2}
    with open(f'plots/{tag}_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'[CNO] Results saved to plots/{tag}_*.csv|json')


if __name__ == '__main__':
    main()
