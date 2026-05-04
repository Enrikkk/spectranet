"""
Unified NSL-model trainer for NS ν=1e-5 on the 850/150/200 split.

One scaffold, six models: LSM, F_FNO, U_FNO, U_NO, MWT, Galerkin_Transformer.
All six share the forward signature forward(x, fx, T=None, geo=None) -> (B, N, out_dim),
so the training loop is identical to ns_transolver.py / ns_factformer.py — only the model
class and its hyperparams change.

Training-protocol invariants (MATCHES existing Transolver/FactFormer/GNOT/ONO baselines):
  * ntrain=850, nval=150, ntest=200, train[:850] / val[850:1000] / test[-200:]
  * T_in=10, T_out=10, step=1, autoregressive with teacher forcing (GT fed back)
  * epochs=200
  * Loss = relative L2 (NSL's L2Loss.rel == Transolver's TestLoss.rel — byte-identical)
  * AdamW + OneCycleLR
  * Best-val ckpt saved + final ckpt; both evaluated on test; per-sample CSV dumped

Run from: ./third_party/NSL/
Expected command-line dispatch (defaults are paper NS settings for each model):
  python3 ns_nsl.py --model_name LSM
  python3 ns_nsl.py --model_name F_FNO
  python3 ns_nsl.py --model_name U_FNO
  python3 ns_nsl.py --model_name U_NO
  python3 ns_nsl.py --model_name MWT
  python3 ns_nsl.py --model_name Galerkin_Transformer
"""
import os, sys, csv, json, random
import scipy.io as scio
import numpy as np
import torch
from tqdm import tqdm
import argparse
from argparse import Namespace

VALID_MODELS = ['LSM', 'F_FNO', 'U_FNO', 'U_NO', 'MWT', 'Galerkin_Transformer', 'FNO', 'U_Net', 'Transformer']

parser = argparse.ArgumentParser('NSL NS baselines — our split')
parser.add_argument('--model_name',   type=str, required=True, choices=VALID_MODELS)
parser.add_argument('--lr',           type=float, default=None, help='if None, use per-model paper default')
parser.add_argument('--epochs',       type=int,   default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--n-hidden',     type=int,   default=None, help='per-model default if None')
parser.add_argument('--n-layers',     type=int,   default=8)
parser.add_argument('--n-heads',      type=int,   default=8)
parser.add_argument('--batch-size',   type=int,   default=None, help='per-model default if None')
parser.add_argument('--gpu',          type=str,   default='0')
parser.add_argument('--max_grad_norm',type=float, default=None)
parser.add_argument('--downsample',   type=int,   default=1)
parser.add_argument('--unified_pos',  type=int,   default=0)
parser.add_argument('--ref',          type=int,   default=8)
parser.add_argument('--modes',        type=int,   default=12)
parser.add_argument('--mlp_ratio',    type=int,   default=1)
parser.add_argument('--dropout',      type=float, default=0.0)
parser.add_argument('--slice_num',    type=int,   default=32)
parser.add_argument('--mwt-k',        type=int,   default=3,  help='MWT wavelet basis k')
parser.add_argument('--act',          type=str,   default='gelu')
parser.add_argument('--save_name',    type=str,   default=None)
parser.add_argument('--data_path',    type=str,   default='./data/NavierStokes_V1e-5_N1200_T20.mat')
parser.add_argument('--resume',       action='store_true', help='Resume from checkpoints/{save_name}_checkpoint.pt')
parser.add_argument('--seed',         type=int,   default=0, help='RNG seed for torch / numpy / random')
args = parser.parse_args()

# ---- per-model default hyperparams (match NSL/scripts/StandardBench/ns/*.sh) ----
MODEL_DEFAULTS = {
    'LSM':                  {'lr': 5e-4,  'n_hidden': 64,  'batch_size': 20, 'unified_pos': 0, 'mlp_ratio': 1},
    'F_FNO':                {'lr': 2.5e-3,'n_hidden': 20,  'batch_size': 20, 'unified_pos': 0, 'mlp_ratio': 2, 'max_grad_norm': 0.1},
    'U_FNO':                {'lr': 5e-4,  'n_hidden': 64,  'batch_size': 20, 'unified_pos': 0, 'mlp_ratio': 1},
    'U_NO':                 {'lr': 1e-3,  'n_hidden': 64,  'batch_size': 16, 'unified_pos': 0, 'mlp_ratio': 1},
    'MWT':                  {'lr': 1e-3,  'n_hidden': 256, 'batch_size': 2,  'unified_pos': 1, 'mlp_ratio': 2},
    'Galerkin_Transformer': {'lr': 1e-3,  'n_hidden': 256, 'batch_size': 2,  'unified_pos': 1, 'mlp_ratio': 2},
    'FNO':                  {'lr': 5e-4,  'n_hidden': 64,  'batch_size': 20, 'unified_pos': 0, 'mlp_ratio': 1},
    'U_Net':                {'lr': 1e-3,  'n_hidden': 256, 'batch_size': 2,  'unified_pos': 1, 'mlp_ratio': 2},
    'Transformer':          {'lr': 1e-3,  'n_hidden': 256, 'batch_size': 2,  'unified_pos': 1, 'mlp_ratio': 2},
}
md = MODEL_DEFAULTS[args.model_name]
if args.lr           is None: args.lr         = md['lr']
if args.n_hidden     is None: args.n_hidden   = md['n_hidden']
if args.batch_size   is None: args.batch_size = md['batch_size']
if args.unified_pos  == 0 and md.get('unified_pos', 0) == 1: args.unified_pos = 1
if args.mlp_ratio    == 1 and md.get('mlp_ratio', 1)    == 2: args.mlp_ratio = 2
if args.max_grad_norm is None and 'max_grad_norm' in md:      args.max_grad_norm = md['max_grad_norm']
if args.save_name    is None:
    base = f'ns_{args.model_name}_our_split'
    args.save_name = f'{base}_s{args.seed}' if args.seed != 0 else base

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Import model classes directly — NSL's model_factory also pulls in GraphSAGE etc. which
# require torch_geometric (not installed on this cluster). Direct imports sidestep that.
import importlib
_MODEL_MODULES = {
    'LSM':                  'models.LSM',
    'F_FNO':                'models.F_FNO',
    'U_FNO':                'models.U_FNO',
    'U_NO':                 'models.U_NO',
    'MWT':                  'models.MWT',
    'Galerkin_Transformer': 'models.Galerkin_Transformer',
    'FNO':                  'models.FNO',
    'U_Net':                'models.U_Net',
    'Transformer':          'models.Transformer',
}
def _load_model(name, model_args):
    mod = importlib.import_module(_MODEL_MODULES[name])
    return mod.Model(model_args)
from utils.loss import L2Loss

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

ntrain, nval, ntest = 850, 150, 200
T_in, T, step = 10, 10, 1
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('plots',       exist_ok=True)


def eval_checkpoint(ckpt_path, model, loader, n_samples, device):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    total, per = 0.0, []
    loss_fn = L2Loss(size_average=False, reduction=False)
    with torch.no_grad():
        for x, fx, yy in loader:
            x, fx, yy = x.to(device), fx.to(device), yy.to(device)
            bsz = x.shape[0]; pred = None
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

    u = scio.loadmat(args.data_path)['u']
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
        gen = torch.Generator().manual_seed(args.seed) if shuffle else None
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos.repeat(n,1,1), a, u),
            batch_size=args.batch_size, shuffle=shuffle, generator=gen)

    train_loader = make_loader(train_a, train_u, ntrain, True)
    val_loader   = make_loader(val_a,   val_u,   nval,   False)
    test_loader  = make_loader(test_a,  test_u,  ntest,  False)

    model_args = Namespace(
        task='dynamic_autoregressive',
        geotype='structured_2D',
        shapelist=[h, h],
        unified_pos=args.unified_pos,
        ref=args.ref,
        fun_dim=T_in,
        space_dim=2,
        out_dim=step,
        n_hidden=args.n_hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        act=args.act,
        time_input=False,
        modes=args.modes,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        slice_num=args.slice_num,
        model=args.model_name,
        mwt_k=args.mwt_k,
    )
    model = _load_model(args.model_name, model_args).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[{args.model_name}] Parameters: {n_params:,}')
    print(f'[{args.model_name}] batch={args.batch_size} lr={args.lr} n_hidden={args.n_hidden} '
          f'unified_pos={args.unified_pos} mlp_ratio={args.mlp_ratio}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    train_loss_fn = L2Loss(size_average=False)
    val_loss_fn   = L2Loss(size_average=False)

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
        print(f'[{args.model_name}] Resumed from epoch {ckpt["epoch"]}, best_val={best_val:.4f}')

    losses_path = f'plots/{args.save_name}_losses.csv'
    csv_mode = 'a' if (args.resume and os.path.exists(ckpt_path)) else 'w'
    losses_file = open(losses_path, csv_mode, newline='')
    losses_writer = csv.writer(losses_file)
    if csv_mode == 'w':
        losses_writer.writerow(['epoch', 'train_l2', 'val_l2'])

    for ep in tqdm(range(start_epoch, args.epochs), desc=f'{args.model_name} NS'):
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

        torch.save({
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'epoch':      ep,
            'best_val':   best_val,
            'best_epoch': best_epoch,
        }, f'checkpoints/{args.save_name}_checkpoint.pt')

        if (ep+1) % 20 == 0 or ep == args.epochs-1:
            print(f'[{args.model_name}] Ep {ep+1:3d}  train={train_l2:.4f}  val={val_l2:.4f}  best={best_val:.4f}@{best_epoch}')

    losses_file.close()
    torch.save(model.state_dict(), f'checkpoints/{args.save_name}_final.pt')
    print(f'\n[{args.model_name}] Best val: epoch {best_epoch}, val L2={best_val:.4f}')

    best_test_l2,  per_best  = eval_checkpoint(f'checkpoints/{args.save_name}_best.pt',  model, test_loader, ntest, device)
    final_test_l2, per_final = eval_checkpoint(f'checkpoints/{args.save_name}_final.pt', model, test_loader, ntest, device)

    print(f'\n[{args.model_name}] Best-val  test L2 = {best_test_l2:.4f}  (epoch {best_epoch})')
    print(f'[{args.model_name}] Final-ep  test L2 = {final_test_l2:.4f}  (epoch {args.epochs})')
    print(f'[{args.model_name}] Overfit gap        = {final_test_l2 - best_test_l2:+.4f}')

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
    print(f'[{args.model_name}] Results saved to plots/{tag}_test_results.csv')


if __name__ == '__main__':
    main()
