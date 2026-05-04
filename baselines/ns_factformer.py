"""
FactFormer on NS ν=1e-5 — adapted to our 850/150/200 split with FNO mat data.

Original: https://github.com/BaratiLab/FactFormer  (examples/turb_ns2d_fact_lm.py)
Changes:
  - Data: NavierStokes_V1e-5_N1200_T20.mat (64×64), not kf_2d_re1000_256_120seed.npy
  - ntrain=850, nval=150, ntest=200
  - epochs=200
  - Curriculum training: ramp latent_steps 1→T_out over first 20 epochs (mirrors original)
  - Autoregressive test rollout in chunks of max_latent_steps (matches original eval)
  - pos_lst uses linspace(0, 2π, h) — matches original step_fn convention
  - Best-val checkpoint + per-sample test CSV output
  - Model config: dim=128, depth=4, heads=8, dim_head=64 (matches paper NS settings)

Run from: ./third_party/FactFormer/
"""
import os, sys, csv, json, math
import scipy.io as scio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from einops import rearrange
from einops.layers.torch import Rearrange

parser = argparse.ArgumentParser('FactFormer NS — our split')
parser.add_argument('--lr',              type=float, default=5e-4)
parser.add_argument('--epochs',          type=int,   default=500)
parser.add_argument('--weight_decay',    type=float, default=1e-5)
parser.add_argument('--dim',             type=int,   default=128)
parser.add_argument('--depth',           type=int,   default=4)
parser.add_argument('--heads',           type=int,   default=8)
parser.add_argument('--dim-head',        type=int,   default=64)
parser.add_argument('--latent-mult',     type=float, default=2.0)
parser.add_argument('--kernel-mult',     type=int,   default=2)
parser.add_argument('--batch-size',      type=int,   default=4)
parser.add_argument('--gpu',             type=str,   default='0')
parser.add_argument('--max_grad_norm',   type=float, default=1.0)
parser.add_argument('--downsample',      type=int,   default=1)
parser.add_argument('--curriculum-epochs', type=int, default=20,
                    help='epochs over which latent_steps is ramped 1→T_out')
parser.add_argument('--save_name',       type=str,   default='ns_factformer_our_split')
parser.add_argument('--data_path',       type=str,   default='./data/NavierStokes_V1e-5_N1200_T20.mat')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libs.factorization_module import FABlock2D
from libs.positional_encoding_module import GaussianFourierFeatureTransform
from libs.basics import PreNorm, MLP

ntrain, nval, ntest = 850, 150, 200
T_in, T_out = 10, 10
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('plots',       exist_ok=True)


# ---------------------------------------------------------------------------
# FactFormer2D (extracted from examples/turb_ns2d_fact_lm.py)
# ---------------------------------------------------------------------------

class FactorizedTransformer(nn.Module):
    def __init__(self, dim, dim_head, heads, dim_out, depth, kernel_multiplier=2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            layer = nn.ModuleList([
                nn.Sequential(
                    GaussianFourierFeatureTransform(2, dim // 2, 8),
                    nn.Linear(dim, dim)
                ),
                FABlock2D(dim, dim_head, dim, heads, dim_out, use_rope=True,
                          kernel_multiplier=kernel_multiplier)
            ])
            self.layers.append(layer)

    def forward(self, u, pos_lst):
        b, nx, ny, c = u.shape
        nx, ny = pos_lst[0].shape[0], pos_lst[1].shape[0]
        pos = torch.stack(
            torch.meshgrid([pos_lst[0].squeeze(-1), pos_lst[1].squeeze(-1)], indexing='ij'),
            dim=-1
        )   # (nx, ny, 2)
        for pos_enc, attn_layer in self.layers:
            u = u + pos_enc(pos.reshape(1, nx * ny, 2)).reshape(1, nx, ny, -1)
            # FABlock2D's domain_wise GroupNorm collapses spatial dims → (B, nx*ny, dim);
            # reshape back to (B, nx, ny, dim) before the residual connection.
            u = attn_layer(u, pos_lst).view(b, nx, ny, -1) + u
        return u


class FactFormer2D(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, in_time_window, out_dim,
                 latent_multiplier=2.0, kernel_multiplier=2, max_latent_steps=10):
        super().__init__()
        self.max_latent_steps = max_latent_steps
        self.latent_dim = int(dim * latent_multiplier)

        self.to_in = nn.Linear(in_time_window, dim, bias=True)
        self.encoder = FactorizedTransformer(dim, dim_head, heads, dim, depth,
                                             kernel_multiplier=kernel_multiplier)
        self.expand_latent = nn.Linear(dim, self.latent_dim, bias=False)
        self.latent_time_emb = nn.Parameter(
            torch.randn(1, max_latent_steps, 1, 1, self.latent_dim) * 0.02,
            requires_grad=True
        )
        self.propagator = PreNorm(
            self.latent_dim,
            MLP([self.latent_dim, dim, self.latent_dim], act_fn=nn.GELU())
        )
        num_groups = int(8 * latent_multiplier)
        assert self.latent_dim % num_groups == 0, \
            f"latent_dim {self.latent_dim} not divisible by num_groups {num_groups}"
        self.simple_to_out = nn.Sequential(
            Rearrange('b nx ny c -> b c (nx ny)'),
            nn.GroupNorm(num_groups=num_groups, num_channels=self.latent_dim),
            nn.Conv1d(self.latent_dim, dim, kernel_size=1, groups=8, bias=False),
            nn.GELU(),
            nn.Conv1d(dim, dim // 2, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv1d(dim // 2, out_dim, kernel_size=1, bias=True)
        )

    def forward(self, u, pos_lst, latent_steps=1):
        """
        u: (B, nx, ny, T_in)
        returns: (B, latent_steps, nx, ny, out_dim)
        """
        b, nx, ny, c = u.shape
        u = self.to_in(u)                      # (B, nx, ny, dim)
        u = self.encoder(u, pos_lst)            # (B, nx, ny, dim)
        u = self.expand_latent(u)               # (B, nx, ny, latent_dim)
        u_lst = []
        for l_t in range(latent_steps):
            u = u + self.latent_time_emb[:, l_t, ...]
            u = self.propagator(u) + u
            u_lst.append(u)
        u = torch.cat(u_lst, dim=0)                         # (latent_steps*B, nx, ny, latent_dim)
        u = self.simple_to_out(u)                            # (latent_steps*B, out_dim, nx*ny)
        u = rearrange(u, '(t b) c (nx ny) -> b t nx ny c',
                      nx=nx, ny=ny, t=latent_steps)          # (B, latent_steps, nx, ny, out_dim)
        return u


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def rel_l2_batch(pred, target):
    """Per-sample relative L2, shape (B,). Inputs can be any shape with B as dim 0."""
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    return (p - t).norm(dim=1) / (t.norm(dim=1) + 1e-8)


def curriculum_latent_steps(ep, curriculum_epochs, T_out):
    """Linear ramp: epoch 0 → 1 step, epoch curriculum_epochs → T_out steps."""
    if curriculum_epochs <= 0:
        return T_out
    progress = min(ep / curriculum_epochs, 1.0)
    return max(1, round(1 + progress * (T_out - 1)))


# ---------------------------------------------------------------------------
# Evaluation helper — autoregressive rollout in chunks of max_latent_steps
# ---------------------------------------------------------------------------

def eval_checkpoint(ckpt_path, model, loader, n_samples, pos_lst, h, device):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    max_ls = model.max_latent_steps
    total = 0.0
    per   = []
    with torch.no_grad():
        for xu, yu in loader:
            xu, yu = xu.to(device), yu.to(device)   # (B, H, W, T_in), (B, H, W, T_out)
            bsz = xu.shape[0]
            n_chunks = math.ceil(T_out / max_ls)
            y_hat = torch.zeros(bsz, T_out, h, h, 1, device=device)
            x_cur = xu
            for i in range(n_chunks):
                steps = min(max_ls, T_out - i * max_ls)
                chunk = model(x_cur, pos_lst, latent_steps=steps)   # (B, steps, H, W, 1)
                y_hat[:, i*max_ls:i*max_ls+steps] = chunk
                # feed predictions back as input (AR rollout)
                new_t = rearrange(chunk[:, :steps], 'b t h w 1 -> b h w t')
                x_cur = torch.cat([x_cur[..., steps:], new_t], dim=-1)
            pred = y_hat.squeeze(-1).permute(0, 2, 3, 1)   # (B, H, W, T_out)
            per_batch = rel_l2_batch(pred, yu)
            per.extend(per_batch.cpu().tolist())
            total += per_batch.sum().item()
    return total / n_samples, per


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    r = args.downsample
    h = int(((64-1)/r)+1)
    device = torch.device('cuda')

    u = scio.loadmat(args.data_path)['u']         # (N, 64, 64, T_total)
    print(f'Loaded: {u.shape}')

    def prep(s):
        xa = torch.from_numpy(u[s, ::r, ::r, :T_in][:, :h, :h, :]).float()          # (N,H,W,T_in)
        xb = torch.from_numpy(u[s, ::r, ::r, T_in:T_in+T_out][:, :h, :h, :]).float()  # (N,H,W,T_out)
        return xa, xb

    train_x, train_y = prep(slice(0, ntrain))
    val_x,   val_y   = prep(slice(ntrain, ntrain+nval))
    test_x,  test_y  = prep(slice(-ntest, None))

    def make_loader(x, y, shuffle):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(train_x, train_y, True)
    val_loader   = make_loader(val_x,   val_y,   False)
    test_loader  = make_loader(test_x,  test_y,  False)

    # pos_lst: coordinates in [0, 2π] — matches original step_fn convention
    pos_lst = [torch.linspace(0, 2*math.pi, h).unsqueeze(-1).to(device),
               torch.linspace(0, 2*math.pi, h).unsqueeze(-1).to(device)]

    model = FactFormer2D(
        dim=args.dim, depth=args.depth, heads=args.heads,
        dim_head=args.dim_head, in_time_window=T_in, out_dim=1,
        latent_multiplier=args.latent_mult, kernel_multiplier=args.kernel_mult,
        max_latent_steps=T_out
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    best_val, best_epoch = float('inf'), -1

    losses_path = f'plots/{args.save_name}_losses.csv'
    losses_file = open(losses_path, 'w', newline='')
    losses_writer = csv.writer(losses_file)
    losses_writer.writerow(['epoch', 'train_l2', 'val_l2'])

    for ep in tqdm(range(args.epochs), desc='FactFormer NS'):
        cur_k = curriculum_latent_steps(ep, args.curriculum_epochs, T_out)
        model.train()
        train_sum = 0.0
        for xu, yu in train_loader:
            xu, yu = xu.to(device), yu.to(device)
            pred = model(xu, pos_lst, latent_steps=cur_k)       # (B, cur_k, H, W, 1)
            pred = pred.squeeze(-1).permute(0, 2, 3, 1)          # (B, H, W, cur_k)
            target = yu[..., :cur_k]                              # (B, H, W, cur_k)
            loss = rel_l2_batch(pred, target).sum()
            train_sum += loss.item()
            optimizer.zero_grad(); loss.backward()
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); scheduler.step()

        # Validation — full T_out steps (no curriculum)
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for xu, yu in val_loader:
                xu, yu = xu.to(device), yu.to(device)
                pred = model(xu, pos_lst, latent_steps=T_out)
                pred = pred.squeeze(-1).permute(0, 2, 3, 1)
                val_total += rel_l2_batch(pred, yu).sum().item()
        val_l2 = val_total / nval

        train_l2 = train_sum / ntrain
        losses_writer.writerow([ep + 1, train_l2, val_l2])
        losses_file.flush()

        if val_l2 < best_val:
            best_val, best_epoch = val_l2, ep+1
            torch.save(model.state_dict(), f'checkpoints/{args.save_name}_best.pt')

        if (ep+1) % 20 == 0 or ep == args.epochs-1:
            print(f'Ep {ep+1:3d}  cur_k={cur_k}  train={train_l2:.4f}  '
                  f'val={val_l2:.4f}  best={best_val:.4f}@{best_epoch}')

    losses_file.close()
    torch.save(model.state_dict(), f'checkpoints/{args.save_name}_final.pt')
    print(f'\nBest val: epoch {best_epoch}, val L2={best_val:.4f}')

    best_test_l2,  per_best  = eval_checkpoint(
        f'checkpoints/{args.save_name}_best.pt',  model, test_loader, ntest, pos_lst, h, device)
    final_test_l2, per_final = eval_checkpoint(
        f'checkpoints/{args.save_name}_final.pt', model, test_loader, ntest, pos_lst, h, device)

    print(f'\nBest-val  test L2 = {best_test_l2:.4f}  (epoch {best_epoch})')
    print(f'Final-ep  test L2 = {final_test_l2:.4f}  (epoch {args.epochs})')
    print(f'Overfit gap        = {final_test_l2 - best_test_l2:+.4f}')
    print(f'(Transolver paper FactFormer NS: 0.1214)')

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
