from collections import defaultdict, deque
import ast
import torch
import pandas as pd
from torch_geometric.data import Data
import numpy as np

def label_components_and_successor(pd_list):
    adj = defaultdict(list)
    labels = set()
    for a, b, c, d in pd_list:
        labels.update([a, b, c, d])
        adj[a].append(c)
        adj[c].append(a)
        adj[b].append(d)
        adj[d].append(b)
    comps = []
    seen = set()
    for x in labels:
        if x in seen:
            continue
        q = deque([x])
        seen.add(x)
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(sorted(comp))
    succ = {}
    for comp in comps:
        comp_set = set(comp)
        m = min(comp)
        for k in comp:
            succ[k] = k + 1 if (k + 1 in comp_set) else m
    return comps, succ

def crossing_port_roles(pd_list, succ):
    port_role = []
    signs = []
    for t in pd_list:
        a, b, c, d = t
        u1, u2 = a, c
        o1, o2 = b, d

        if succ[u1] == u2:
            in_under, out_under = u1, u2
        elif succ[u2] == u1:
            in_under, out_under = u2, u1
        else:
            raise ValueError(f'Cannot orient under strand for {t}')

        if succ[o1] == o2:
            in_over, out_over = o1, o2
        elif succ[o2] == o1:
            in_over, out_over = o2, o1
        else:
            raise ValueError(f'Cannot orient over strand for {t}')

        pos_in_under = t.index(in_under)
        pos_out_under = t.index(out_under)
        pos_in_over = t.index(in_over)
        pos_out_over = t.index(out_over)

        roles = [None] * 4
        roles[pos_in_under] = 0
        roles[pos_in_over] = 1
        roles[pos_out_under] = 2
        roles[pos_out_over] = 3
        port_role.append(roles)

        sgn = +1 if ((pos_in_over - pos_in_under) % 4 == 1) else -1
        signs.append(sgn)

    return port_role, np.array(signs, dtype=np.int64)


def smooth_crossing(pd_list, crossing_idx):
    """Port-role based oriented smoothing."""
    n = len(pd_list)
    if n <= 1: return None, 0
    _, succ = label_components_and_successor(pd_list)
    pr, _ = crossing_port_roles(pd_list, succ)
    t = pd_list[crossing_idx]
    roles = pr[crossing_idx]
    p = {r: i for i, r in enumerate(roles)}
    sub = {t[p[0]]: t[p[3]], t[p[1]]: t[p[2]]}
    new_pd = [[sub.get(a, a) for a in tt] for i, tt in enumerate(pd_list) if i != crossing_idx]
    if not new_pd: return None, 0
    arcs = sorted(set(a for tt in new_pd for a in tt))
    rl = {a: i+1 for i, a in enumerate(arcs)}
    new_pd = [[rl[a] for a in tt] for tt in new_pd]
    adj = defaultdict(set)
    for aa, bb, cc, dd in new_pd:
        adj[aa].add(cc); adj[cc].add(aa); adj[bb].add(dd); adj[dd].add(bb)
    seen = set(); nc = 0
    for s in adj:
        if s in seen: continue
        nc += 1; stk = [s]
        while stk:
            u = stk.pop()
            if u in seen: continue
            seen.add(u); stk.extend(v for v in adj[u] if v not in seen)
    return new_pd, nc



# graph G_0 described in report
def build_designA_graph(pd_list):
    _, succ = label_components_and_successor(pd_list)
    port_role, signs = crossing_port_roles(pd_list, succ)

    occ = defaultdict(list)
    for i, t in enumerate(pd_list):
        for pos, lab in enumerate(t):
            occ[int(lab)].append((i, port_role[i][pos]))

    srcs, dsts, src_port, dst_port, edge_type = [], [], [], [], []
    for lab, endpoints in occ.items():
        if len(endpoints) != 2:
            raise ValueError(f'Arc label {lab} occurs {len(endpoints)} times (expected 2)')
        (i, ri), (j, rj) = endpoints

        is_out = lambda r: r in (2, 3)
        is_in = lambda r: r in (0, 1)

        if is_out(ri) and is_in(rj):
            s, t = i, j
            sp, tp = ri, rj
        elif is_out(rj) and is_in(ri):
            s, t = j, i
            sp, tp = rj, ri
        else:
            raise ValueError(f'Arc label {lab} does not connect Out->In: {endpoints}')

        srcs.append(s)
        dsts.append(t)
        src_port.append(sp)
        dst_port.append(tp)

        src_is_over = 1 if sp == 3 else 0
        dst_is_over = 1 if tp == 1 else 0
        edge_type.append(2 * src_is_over + dst_is_over)

    x = torch.tensor(np.stack([signs, np.ones_like(signs)], axis=1), dtype=torch.float)
    data = Data(
        x=x,
        edge_index=torch.tensor([srcs, dsts], dtype=torch.long),
        edge_type=torch.tensor(edge_type, dtype=torch.long),
    )
    data.src_port = torch.tensor(src_port, dtype=torch.long)
    data.dst_port = torch.tensor(dst_port, dtype=torch.long)
    return data

# graph G
def to_strand_split_graph(data, add_coupling=True):
    N = data.num_nodes
    src, dst = data.edge_index
    sp = data.src_port
    dp = data.dst_port

    src2 = torch.where(sp == 2, src, src + N)
    dst2 = torch.where(dp == 0, dst, dst + N)

    edge_index = torch.stack([src2, dst2], dim=0)
    edge_type = data.edge_type.clone()

    sign = data.x[:, 0:1]
    one = data.x[:, 1:2]
    under_flag = torch.zeros((N, 1), dtype=data.x.dtype)
    over_flag = torch.ones((N, 1), dtype=data.x.dtype)

    x_under = torch.cat([sign, one, under_flag], dim=1)
    x_over = torch.cat([sign, one, over_flag], dim=1)
    x = torch.cat([x_under, x_over], dim=0)

    if add_coupling:
        u = torch.arange(N, dtype=torch.long)
        cross_src = torch.cat([u, u + N])
        cross_dst = torch.cat([u + N, u])
        cross_edge_index = torch.stack([cross_src, cross_dst], dim=0)
        coupling_rel = edge_type.max().item() + 1
        cross_edge_type = torch.full((2 * N,), coupling_rel, dtype=edge_type.dtype)
        edge_index = torch.cat([edge_index, cross_edge_index], dim=1)
        edge_type = torch.cat([edge_type, cross_edge_type], dim=0)

    return Data(x=x, edge_index=edge_index, edge_type=edge_type)

def add_reverse_edges(data, num_relations):
    ei = data.edge_index
    et = data.edge_type
    rev_ei = ei.flip(0)
    rev_et = et + num_relations
    data.edge_index = torch.cat([ei, rev_ei], dim=1)
    data.edge_type = torch.cat([et, rev_et], dim=0)
    return data