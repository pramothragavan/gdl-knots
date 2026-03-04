import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_add_pool
import numpy as np

class EncoderV2(nn.Module):
    def __init__(self, in_dim, hidden=128, num_relations=10, num_layers=6, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RGCNConv(hidden, hidden, num_relations=num_relations))
            self.norms.append(nn.LayerNorm(hidden))
        self.dropout = dropout

    def forward(self, batch):
        x = self.input_proj(batch.x)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, batch.edge_index, batch.edge_type)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual  # skip connection
        return global_add_pool(x, batch.batch)

class HurdleEncoderV1(nn.Module):
    def __init__(self, in_dim, out_dim, value_mode='regression', num_value_classes=None):
        super().__init__()
        self.value_mode = value_mode
        self.encoder = EncoderV2(in_dim=in_dim)
        self.mask_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, out_dim))

        if value_mode == 'classification':
            assert num_value_classes is not None and num_value_classes > 1
            self.value_head = nn.Sequential(
                nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, out_dim * num_value_classes)
            )
            self.num_value_classes = num_value_classes
            self.out_dim = out_dim
        else:
            self.value_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, out_dim))

    def forward(self, batch):
        g = self.encoder(batch)
        logits = self.mask_head(g)
        value_raw = self.value_head(g)
        if self.value_mode == 'classification':
            value_raw = value_raw.view(-1, self.out_dim, self.num_value_classes)
        return logits, value_raw


def rank_auc(prob_flat, mask_flat):
    y = mask_flat.float()
    x = prob_flat.float()
    pos = y.sum().item()
    neg = (1 - y).sum().item()
    if pos == 0 or neg == 0:
        return float('nan')
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(x) + 1, dtype=torch.float)
    sum_ranks_pos = ranks[y == 1].sum().item()
    auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def f1_at_threshold(prob_flat, mask_flat, thr):
    pred = (prob_flat >= thr).long()
    y = mask_flat.long()
    tp = int(((pred == 1) & (y == 1)).sum().item())
    fp = int(((pred == 1) & (y == 0)).sum().item())
    fn = int(((pred == 0) & (y == 1)).sum().item())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def tune_threshold(prob_flat, mask_flat):
    thresholds = [round(float(t), 2) for t in np.arange(0.05, 0.951, 0.05)]
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_at_threshold(prob_flat, mask_flat, t)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

def build_value_codec(K, max_abs, use_overflow=True):
    K_eff = int(min(K, max_abs if max_abs > 0 else K))
    base_vals = list(range(-K_eff, 0)) + list(range(1, K_eff + 1))
    if use_overflow:
        overflow_neg = -(K_eff + 1)
        overflow_pos = +(K_eff + 1)
        class_values = base_vals + [overflow_neg, overflow_pos]
    else:
        class_values = base_vals
    v2i = {v: i for i, v in enumerate(class_values)}
    return {'K': K_eff, 'use_overflow': use_overflow, 'class_values': class_values, 'value_to_idx': v2i}

def value_to_class_tensor(y_nonzero, codec):
    vals = y_nonzero.round().long().clone()
    K = int(codec['K'])
    if codec['use_overflow']:
        vals = torch.where(vals > K, torch.full_like(vals, K + 1), vals)
        vals = torch.where(vals < -K, torch.full_like(vals, -(K + 1)), vals)
    else:
        vals = torch.clamp(vals, min=-K, max=K)
    idx = [codec['value_to_idx'][int(v.item())] for v in vals]
    return torch.tensor(idx, dtype=torch.long, device=y_nonzero.device)


def decode_values(value_raw, value_mode, codec=None, decode_mode='expected'):
    if value_mode == 'regression':
        return value_raw
    class_values = torch.tensor(codec['class_values'], dtype=value_raw.dtype, device=value_raw.device)
    probs = F.softmax(value_raw, dim=-1)
    if decode_mode == 'argmax':
        cls = probs.argmax(dim=-1)
        return class_values[cls]
    return (probs * class_values.view(1, 1, -1)).sum(dim=-1)


@torch.no_grad()
def evaluate_hurdle(model, loader, device, value_mode='regression', codec=None, decode_mode='expected'):
    model.eval()
    all_prob, all_mask = [], []
    all_values_pred, all_y = [], []

    for b in loader:
        b = b.to(device)
        y = b.y.float().view(b.num_graphs, -1)
        logits, value_raw = model(b)
        p = torch.sigmoid(logits)
        values_pred = decode_values(value_raw, value_mode=value_mode, codec=codec, decode_mode=decode_mode)

        all_prob.append(p.detach().cpu())
        all_values_pred.append(values_pred.detach().cpu())
        all_mask.append((y != 0).detach().cpu())
        all_y.append(y.detach().cpu())

    PROB = torch.cat(all_prob, dim=0)
    VALUES = torch.cat(all_values_pred, dim=0)
    MASK = torch.cat(all_mask, dim=0)
    Y = torch.cat(all_y, dim=0)

    prob_flat = PROB.reshape(-1)
    mask_flat = MASK.reshape(-1)
    best_t, best_f1 = tune_threshold(prob_flat, mask_flat)
    auc = rank_auc(prob_flat, mask_flat)

    pred_soft = PROB * VALUES
    hard_mask = (PROB >= best_t).float()
    pred_hard = hard_mask * VALUES
    pred_oracle = MASK.float() * VALUES

    nz = MASK
    mae_all_soft = float((pred_soft - Y).abs().mean().item())
    mae_nz_soft = float((pred_soft[nz] - Y[nz]).abs().mean().item()) if nz.any() else float('nan')
    mae_nz_hard = float((pred_hard[nz] - Y[nz]).abs().mean().item()) if nz.any() else float('nan')
    mae_nz_oracle_gate = float((pred_oracle[nz] - Y[nz]).abs().mean().item()) if nz.any() else float('nan')
    mae_values_on_true_nz = float((VALUES[nz] - Y[nz]).abs().mean().item()) if nz.any() else float('nan')

    pred_pos = hard_mask.bool()
    fn = ((~pred_pos) & nz).sum().item()
    fp = (pred_pos & (~nz)).sum().item()
    n_pos = nz.sum().item()
    n_neg = (~nz).sum().item()
    gate_fn_rate = float(fn / max(n_pos, 1))
    gate_fp_rate = float(fp / max(n_neg, 1))

    out = {
        'val_MAE_all_soft': mae_all_soft,
        'val_MAE_nz_soft': mae_nz_soft,
        'val_MAE_nz_hard': mae_nz_hard,
        'val_MAE_nz_oracle_gate': mae_nz_oracle_gate,
        'val_mae_values_on_true_nz': mae_values_on_true_nz,
        'mask_auc': auc,
        'mask_f1': float(best_f1),
        'best_thresh': float(best_t),
        'gate_fn_rate': gate_fn_rate,
        'gate_fp_rate': gate_fp_rate,
        'val_pred_abs_mean': float(pred_soft.abs().mean().item()),
    }

    if value_mode == 'classification':
        exact = (VALUES[nz].round() == Y[nz].round()).float().mean().item() if nz.any() else float('nan')
        out['val_value_exact_acc_nz'] = float(exact)

    return out

def compute_pos_weight(train_loader, L):
    n_pos = torch.zeros(L, dtype=torch.float)
    n_samples = 0
    for b in train_loader:
        y = b.y.float().view(b.num_graphs, -1)
        m = (y != 0).float()
        n_pos += m.sum(dim=0)
        n_samples += y.size(0)
    n_neg = float(n_samples) - n_pos
    pos_weight = n_neg / torch.clamp(n_pos, min=1.0)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=50.0)
    return pos_weight

def build_window_mask(y, pad=None):
    m = (y != 0).float()
    if pad is None:
        return m
    B, Ldim = y.shape
    w = torch.zeros_like(m)
    for i in range(B):
        idx = torch.where(m[i] > 0)[0]
        if idx.numel() == 0:
            continue
        lo = max(0, int(idx.min().item()) - int(pad))
        hi = min(Ldim - 1, int(idx.max().item()) + int(pad))
        w[i, lo:hi + 1] = 1.0
    return w


def hurdle_losses(logits, value_raw, y, pos_weight, lam,
                  value_mode='regression', codec=None, decode_mode='expected',
                  window_pad=None, teacher_forcing_reg=False):
    m = (y != 0).float()
    bce_per = F.binary_cross_entropy_with_logits(
        logits, m, pos_weight=pos_weight.view(1, -1), reduction='none')
    bce = bce_per.mean(dim=1).mean()

    if value_mode == 'classification':
        nz = (m > 0)
        if nz.any():
            target_cls = value_to_class_tensor(y[nz], codec)
            ce = F.cross_entropy(value_raw[nz], target_cls, reduction='mean')
        else:
            ce = torch.zeros((), device=y.device, dtype=y.dtype)
        reg = ce
    else:
        if teacher_forcing_reg:
            pred_reg = m * value_raw
        else:
            pred_reg = torch.sigmoid(logits) * value_raw
        reg_mask = build_window_mask(y, pad=window_pad)
        reg_per = F.smooth_l1_loss(pred_reg, y, reduction='none')
        denom = reg_mask.sum(dim=1).clamp(min=1.0)
        reg = (reg_per * reg_mask).sum(dim=1) / denom
        reg = reg.mean()

    total = bce + lam * reg
    return total, bce, reg


def collect_nonzero_values(train_loader):
    vals = []
    for b in train_loader:
        y = b.y.float().view(b.num_graphs, -1)
        nz = y[y != 0].cpu().numpy().astype(int)
        vals.extend(nz.tolist())
    return vals