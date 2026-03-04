import pandas as pd
import ast
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from utils import build_designA_graph, to_strand_split_graph, add_reverse_edges
import os

POLY_KINDS = ['Conway']
def conway_fold(y_full, min_exp):
    """Extract even-exponent coefficients from Conway dense vector."""
    L = y_full.shape[-1]
    even_idx = [i for i in range(L) if (min_exp + i) % 2 == 0]
    even_idx_t = torch.tensor(even_idx, dtype=torch.long)
    return y_full[..., even_idx_t], even_idx_t


def conway_unfold(y_even, L, even_indices):
    """Reconstruct full Conway vector from even-exponent coefficients."""
    shape = list(y_even.shape[:-1]) + [L]
    y_full = torch.zeros(shape, dtype=y_even.dtype, device=y_even.device)
    y_full[..., even_indices] = y_even
    return y_full

def verify_conway_even(y_full, min_exp, tol=1e-4):
    """Check what fraction of odd-exponent entries are zero."""
    L = y_full.shape[-1]
    odd_idx = [i for i in range(L) if (min_exp + i) % 2 != 0]
    if not odd_idx:
        return 1.0
    odd_vals = y_full[..., torch.tensor(odd_idx, dtype=torch.long)]
    return (odd_vals.abs() < tol).float().mean().item()

def determinant_from_poly_dict(poly_dict):
    """Compute |Delta(-1)| from a polynomial dict {exp: coeff}."""
    val = sum(c * ((-1) ** e) for e, c in poly_dict.items())
    return abs(val)

def parse_pd_semicolon(s: str):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t:
        return None
    try:
        obj = ast.literal_eval(t.replace(';', ','))
        out = []
        for row in obj:
            if len(row) != 4:
                return None
            out.append([int(x) for x in row])
        return out
    except Exception:
        return None

def split_terms_top_level(expr: str):
    s = expr.replace(' ', '')
    s = s.replace('{', '(').replace('}', ')')
    if s == '':
        return []
    terms = []
    cur = ''
    depth = 0
    for i, ch in enumerate(s):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(depth - 1, 0)
        if ch in '+-' and i > 0 and depth == 0 and s[i - 1] != '^':
            terms.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        terms.append(cur)
    return [t for t in terms if t and t != '+']

def parse_univariate_poly(poly_str, var='t'):
    if poly_str is None or (isinstance(poly_str, float) and pd.isna(poly_str)):
        return None
    if not isinstance(poly_str, str):
        return None
    s = poly_str.strip()
    if s == '':
        return None
    s = s.replace(' ', '')

    terms = split_terms_top_level(s)
    if not terms:
        return None

    out = {}
    for term in terms:
        if term == '':
            continue
        if var in term:
            i = term.find(var)
            coeff_part = term[:i]
            coeff_part = coeff_part[:-1] if coeff_part.endswith('*') else coeff_part
            if coeff_part in ('', '+'):
                coeff = 1
            elif coeff_part == '-':
                coeff = -1
            else:
                coeff = int(coeff_part)
            exp = 1
            rest = term[i + len(var):]
            if rest:
                if not rest.startswith('^'):
                    return None
                exp_txt = rest[1:]
                if exp_txt.startswith('(') and exp_txt.endswith(')'):
                    exp_txt = exp_txt[1:-1]
                exp = int(exp_txt)
        else:
            coeff = int(term)
            exp = 0
        out[exp] = out.get(exp, 0) + coeff

    out = {e: c for e, c in out.items() if c != 0}
    if len(out) == 0:
        out = {0: 0}
    return out

def poly_dict_to_dense(poly_dict, min_exp, max_exp):
    y = torch.zeros(max_exp - min_exp + 1, dtype=torch.float)
    for e, c in poly_dict.items():
        if min_exp <= e <= max_exp:
            y[e - min_exp] = float(c)
    return y


def prepare_poly_dataset(poly_kind='Alexander', split_seed=0, use_folding=True):
    assert poly_kind in ('Alexander', 'Conway') # i'm no longer using alexander, but legacy
    dfp = pd.read_csv('alexconwayjones.csv')
    dfp['pd_list'] = dfp['PD Notation'].apply(parse_pd_semicolon)
    var = 't' if poly_kind == 'Alexander' else 'z'
    dfp[f'{poly_kind}_dict'] = dfp[poly_kind].apply(lambda s: parse_univariate_poly(s, var=var))

    dfp = dfp[dfp['pd_list'].notna() & dfp[f'{poly_kind}_dict'].notna()].copy().reset_index(drop=True)

    perm = np.random.RandomState(split_seed).permutation(len(dfp))
    n_train = int(0.9 * len(dfp))
    tr_idx = perm[:n_train]
    va_idx = perm[n_train:]
    tr_df = dfp.iloc[tr_idx].copy().reset_index(drop=True)
    va_df = dfp.iloc[va_idx].copy().reset_index(drop=True)

    train_exps = []
    for d in tr_df[f'{poly_kind}_dict']:
        train_exps.extend(list(d.keys()))
    min_exp = int(min(train_exps))
    max_exp = int(max(train_exps))
    L_full = max_exp - min_exp + 1

    # Compute scalar targets
    dfp['determinant'] = dfp[f'{poly_kind}_dict'].apply(
        lambda d: determinant_from_poly_dict(d) if d is not None else np.nan
    )
    has_signature = 'Signature' in dfp.columns or 'signature' in dfp.columns
    sig_col = 'Signature' if 'Signature' in dfp.columns else ('signature' if 'signature' in dfp.columns else None)

    print(f"[{poly_kind}] range=[{min_exp},{max_exp}] L_full={L_full}")
    print(f"  has_signature={has_signature}")
    print(f"  determinant range: {dfp['determinant'].min():.0f} to {dfp['determinant'].max():.0f}")

    # Build full dense targets for all rows
    all_y_full = []
    for _, row in dfp.iterrows():
        y = poly_dict_to_dense(row[f'{poly_kind}_dict'], min_exp, max_exp)
        all_y_full.append(y)
    Y_full = torch.stack(all_y_full, dim=0)

    fold_info = {'type': 'none', 'L_full': L_full, 'L_folded': L_full}
    Y_target = Y_full

    if use_folding and poly_kind == 'Conway':
        even_frac = verify_conway_even(Y_full, min_exp)
        print(f"  Even-degree verification: {even_frac*100:.1f}% of odd-exponent entries are zero")

        if even_frac > 0.95:
            Y_even, even_indices = conway_fold(Y_full, min_exp)
            L_out = Y_even.shape[-1]
            fold_info = {
                'type': 'conway_even', 'L_full': L_full, 'L_folded': L_out,
                'min_exp': min_exp, 'even_indices': even_indices,
            }
            Y_target = Y_even
            print(f"  -> Conway folded: L={L_full} -> L_folded={L_out} (even-exponent only)")
        else:
            print(f"  -> Conway even-degree structure not strong enough. Using full vector.")
            Y_target = Y_full

    L_out = fold_info.get('L_folded', L_full)

    # Build PyG graphs
    cache_tag = fold_info['type']
    cache_path = f'alexconway_{poly_kind.lower()}_{min_exp}_{max_exp}_{cache_tag}.pt'

    graphs = None
    if os.path.exists(cache_path):
        try:
            payload = torch.load(cache_path, weights_only=False)
            if payload.get('n') == len(dfp) and payload.get('fold_type') == fold_info['type']:
                graphs = payload['graphs']
                fold_info = payload.get('fold_info', fold_info)
                print(f"  Loaded {len(graphs)} cached graphs ({cache_tag})")
        except Exception:
            pass

    if graphs is None:
        print(f"  Building {len(dfp)} graphs...")
        graphs = []
        for i, (_, row) in enumerate(dfp.iterrows()):
            g = build_designA_graph(row['pd_list'])
            g = to_strand_split_graph(g, add_coupling=True)
            g = add_reverse_edges(g, num_relations=5)
            g.y = Y_target[i].unsqueeze(0)
            g.y_full = Y_full[i].unsqueeze(0)
            g.determinant = torch.tensor([dfp.iloc[i]['determinant']], dtype=torch.float)
            if sig_col is not None and not pd.isna(dfp.iloc[i].get(sig_col, np.nan)):
                g.signature = torch.tensor([float(dfp.iloc[i][sig_col])], dtype=torch.float)
            else:
                g.signature = torch.tensor([float('nan')], dtype=torch.float)
            graphs.append(g)
        torch.save({'n': len(dfp), 'graphs': graphs, 'fold_info': fold_info,
                     'fold_type': fold_info['type']}, cache_path)
        print(f"  Cached to {cache_path}")

    train_graphs = [graphs[i] for i in tr_idx]
    val_graphs = [graphs[i] for i in va_idx]
    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=128, shuffle=False)

    return {
        'poly_kind': poly_kind,
        'df': dfp,
        'train_df': tr_df,
        'val_df': va_df,
        'train_graphs': train_graphs,
        'val_graphs': val_graphs,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'min_exp': min_exp,
        'max_exp': max_exp,
        'L': L_out,
        'L_full': L_full,
        'var': var,
        'folded': use_folding and fold_info['type'] not in ('none', 'none_support_palindromic'),
        'fold_info': fold_info,
        'has_signature': has_signature,
        'sig_col': sig_col,
    }