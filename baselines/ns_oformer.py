"""
OFormer (Operator Transformer) on NS ν=1e-5 — adapted to our 850/150/200 split.

Original: https://github.com/BaratiLab/OFormer  (uniform_grids/tune_navier_stokes.py)
Changes:
  - ntrain=850, nval=150, ntest=200
  - Data: NavierStokes_V1e-5_N1200_T20.mat (.mat scipy format, 64×64)
  - epochs=200, OneCycleLR
  - Autoregressive rollout step=1 over T_out=10, teacher forcing during train
  - Same relative-L2 metric as all other baselines
  - Best-val ckpt + per-sample test CSV output

Run from: ./third_party/OFormer/uniform_grids/
"""
import os, sys, csv, json
import scipy.io as scio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from einops import rearrange, repeat

parser = argparse.ArgumentParser('OFormer NS — our split')
parser.add_argument('--lr',              type=float, default=1e-4)
parser.add_argument('--epochs',          type=int,   default=500)
parser.add_argument('--weight_decay',    type=float, default=1e-5)
parser.add_argument('--in-channels',     type=int,   default=3,   help='T_in + 2 pos channels')
parser.add_argument('--in-emb-dim',      type=int,   default=128, help='encoder embedding dim')
parser.add_argument('--out-seq-emb-dim', type=int,   default=64,  help='latent sequence dim')
parser.add_argument('--encoder-heads',   type=int,   default=1)
parser.add_argument('--encoder-depth',   type=int,   default=5)
parser.add_argument('--decoder-emb-dim', type=int,   default=64)
parser.add_argument('--out-channels',    type=int,   default=1)
parser.add_argument('--out-step',        type=int,   default=1)
parser.add_argument('--propagator-depth',type=int,   default=1)
parser.add_argument('--fourier-frequency',type=int,  default=8)
parser.add_argument('--batch-size',      type=int,   default=8)
parser.add_argument('--gpu',             type=str,   default='0')
parser.add_argument('--max_grad_norm',   type=float, default=1.0)
parser.add_argument('--save_name',       type=str,   default='ns_oformer_our_split')
parser.add_argument('--data_path',       type=str,   default='./data/NavierStokes_V1e-5_N1200_T20.mat')
parser.add_argument('--resume',          action='store_true', help='Resume from checkpoints/{save_name}_checkpoint.pt')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn_module.encoder_module import SpatialTemporalEncoder2D
from nn_module.decoder_module import PointWiseDecoder2D

torch.manual_seed(0)
np.random.seed(0)

ntrain, nval, ntest = 850, 150, 200
T_in, T, step = 10, 10, 1
H = W = 64
N = H * W
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('plots',       exist_ok=True)


def rel_l2_batch(pred, target):
    B = pred.shape[0]
    diff = (pred - target).reshape(B, -1).norm(dim=1)
    norm = target.reshape(B, -1).norm(dim=1)
    return diff / (norm + 1e-8)


def build_pos_grid(h, w, device):
    xs = torch.linspace(0, 1, w, device=device)
    ys = torch.linspace(0, 1, h, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    pos = torch.stack([xx.ravel(), yy.ravel()], dim=-1)  # (N, 2)
    return pos


def make_encoder_input(fx, pos):
    """
    fx:  (B, N, T_in)  — vorticity channels flattened
    pos: (N, 2)         — spatial position

    SpatialTemporalEncoder2D.to_embedding is nn.Linear applied to the last dim,
    so input must be (B, N, T_in+2) — NOT transposed to (B, T_in+2, N).
    """
    B, _, _ = fx.shape
    pos_bc = pos.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
    return torch.cat([fx, pos_bc], dim=-1)        # (B, N, T_in+2)


def eval_checkpoint(ckpt_path, encoder, decoder, loader, n_samples, device, pos):
    state = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])
    encoder.eval(); decoder.eval()
    total, per = 0.0, []
    with torch.no_grad():
        for fx, yy in loader:
            fx, yy = fx.to(device), yy.to(device)  # (B,N,T_in), (B,N,T)
            bsz = fx.shape[0]
            pos_bc = pos.unsqueeze(0).expand(bsz, -1, -1)
            pred_seq = []
            cur_fx = fx.clone()
            for t in range(T):
                enc_in = make_encoder_input(cur_fx, pos)     # (B, N, T_in+2)
                z = encoder(enc_in, pos_bc)                  # (B, N, latent)
                u, _ = decoder(z, pos_bc)                    # u: (B, 1, N)
                out = u.permute(0, 2, 1)                     # (B, N, 1)
                pred_seq.append(out)
                cur_fx = torch.cat([cur_fx[..., step:], out], dim=-1)
            pred = torch.cat(pred_seq, dim=-1)               # (B, N, T)
            per_batch = rel_l2_batch(pred.reshape(bsz,-1), yy.reshape(bsz,-1))
            per.extend(per_batch.cpu().tolist())
            total += per_batch.sum().item()
    return total / n_samples, per


def main():
    device = torch.device('cuda')

    u = scio.loadmat(args.data_path)['u']  # (1200, 64, 64, 20)
    u = torch.from_numpy(u).float()

    def prep(s):
        # flatten spatial → (N, H*W, T)
        a = u[s, :, :, :T_in].reshape(-1, N, T_in)
        b = u[s, :, :, T_in:T_in+T].reshape(-1, N, T)
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

    pos = build_pos_grid(H, W, device)  # (N, 2)

    in_channels = T_in + 2  # T_in vorticity channels + x,y position
    encoder = SpatialTemporalEncoder2D(
        in_channels, args.in_emb_dim, args.out_seq_emb_dim,
        args.encoder_heads, args.encoder_depth).to(device)
    decoder = PointWiseDecoder2D(
        args.decoder_emb_dim, args.out_channels, args.out_step,
        args.propagator_depth, scale=args.fourier_frequency, dropout=0.0).to(device)

    n_params = (sum(p.numel() for p in encoder.parameters() if p.requires_grad) +
                sum(p.numel() for p in decoder.parameters() if p.requires_grad))
    print(f'[OFormer] Parameters: {n_params:,}')

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    best_val, best_epoch = float('inf'), -1
    start_epoch = 0
    ckpt_path = f'checkpoints/{args.save_name}_checkpoint.pt'
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt['best_val']
        best_epoch  = ckpt['best_epoch']
        print(f'[OFormer] Resumed from epoch {ckpt["epoch"]}, best_val={best_val:.4f}')

    losses_path = f'plots/{args.save_name}_losses.csv'
    csv_mode = 'a' if (args.resume and os.path.exists(ckpt_path)) else 'w'
    losses_file = open(losses_path, csv_mode, newline='')
    losses_writer = csv.writer(losses_file)
    if csv_mode == 'w':
        losses_writer.writerow(['epoch', 'train_l2', 'val_l2'])

    for ep in tqdm(range(start_epoch, args.epochs), desc='OFormer NS'):
        encoder.train(); decoder.train()
        train_full = 0.0
        for fx, yy in train_loader:
            fx, yy = fx.to(device), yy.to(device)
            bsz = fx.shape[0]
            pos_bc = pos.unsqueeze(0).expand(bsz, -1, -1)
            cur_fx = fx.clone()
            loss = torch.tensor(0.0, device=device)
            pred_seq = []
            for t in range(T):
                y_gt = yy[..., t:t+step]                     # (B, N, 1)
                enc_in = make_encoder_input(cur_fx, pos)
                z   = encoder(enc_in, pos_bc)
                u, _ = decoder(z, pos_bc)                    # u: (B, 1, N)
                out = u.permute(0, 2, 1)                     # (B, N, 1)
                loss = loss + rel_l2_batch(out.reshape(bsz,-1), y_gt.reshape(bsz,-1)).mean()
                pred_seq.append(out)
                cur_fx = torch.cat([cur_fx[..., step:], y_gt], dim=-1)
            pred = torch.cat(pred_seq, dim=-1)
            train_full += rel_l2_batch(pred.reshape(bsz,-1), yy.reshape(bsz,-1)).sum().item()
            optimizer.zero_grad(); loss.backward()
            if args.max_grad_norm:
                nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step(); scheduler.step()

        encoder.eval(); decoder.eval()
        val_total = 0.0
        with torch.no_grad():
            for fx, yy in val_loader:
                fx, yy = fx.to(device), yy.to(device)
                bsz = fx.shape[0]
                pos_bc = pos.unsqueeze(0).expand(bsz, -1, -1)
                cur_fx = fx.clone()
                pred_seq = []
                for t in range(T):
                    enc_in = make_encoder_input(cur_fx, pos)
                    z   = encoder(enc_in, pos_bc)
                    u, _ = decoder(z, pos_bc)                # u: (B, 1, N)
                    out = u.permute(0, 2, 1)                 # (B, N, 1)
                    pred_seq.append(out)
                    cur_fx = torch.cat([cur_fx[..., step:], out], dim=-1)
                pred = torch.cat(pred_seq, dim=-1)
                val_total += rel_l2_batch(pred.reshape(bsz,-1), yy.reshape(bsz,-1)).sum().item()
        val_l2   = val_total / nval
        train_l2 = train_full / ntrain

        losses_writer.writerow([ep+1, train_l2, val_l2]); losses_file.flush()

        if val_l2 < best_val:
            best_val, best_epoch = val_l2, ep+1
            torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},
                       f'checkpoints/{args.save_name}_best.pt')

        torch.save({
            'encoder':    encoder.state_dict(),
            'decoder':    decoder.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'epoch':      ep,
            'best_val':   best_val,
            'best_epoch': best_epoch,
        }, f'checkpoints/{args.save_name}_checkpoint.pt')

        if (ep+1) % 20 == 0 or ep == args.epochs-1:
            print(f'[OFormer] Ep {ep+1:3d}  train={train_l2:.4f}  val={val_l2:.4f}  best={best_val:.4f}@{best_epoch}')

    losses_file.close()
    torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},
               f'checkpoints/{args.save_name}_final.pt')

    best_test_l2,  per_best  = eval_checkpoint(f'checkpoints/{args.save_name}_best.pt',
                                                encoder, decoder, test_loader, ntest, device, pos)
    final_test_l2, per_final = eval_checkpoint(f'checkpoints/{args.save_name}_final.pt',
                                                encoder, decoder, test_loader, ntest, device, pos)

    print(f'\n[OFormer] Best-val  test L2 = {best_test_l2:.4f}  (epoch {best_epoch})')
    print(f'[OFormer] Final-ep  test L2 = {final_test_l2:.4f}  (epoch {args.epochs})')
    print(f'[OFormer] Overfit gap        = {final_test_l2 - best_test_l2:+.4f}')

    tag = args.save_name
    with open(f'plots/{tag}_test_results.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['sample_idx', 'test_l2_best', 'test_l2_final'])
        for i, (b, fin) in enumerate(zip(per_best, per_final)):
            w.writerow([i, b, fin])

    cfg = {**vars(args), 'model_name': 'OFormer', 'ntrain': ntrain, 'nval': nval, 'ntest': ntest,
           'T_in': T_in, 'T_out': T, 'step': step,
           'n_params': n_params, 'best_epoch': best_epoch,
           'best_test_l2': best_test_l2, 'final_test_l2': final_test_l2}
    with open(f'plots/{tag}_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'[OFormer] Results saved to plots/{tag}_test_results.csv')


if __name__ == '__main__':
    main()
