# a slightly DIY implementation that follows a standard backbone, that's slightly faster for our purposes

import sympy as sp
import numpy as np
import torch
from utils import build_designA_graph
from interp import build_graph_for_inference

def det_bareiss_int(M):
    return int(sp.Matrix(M).det(method="bareiss"))

def det_values_seifert(V_input, t_points):
    """
    Evaluate det(V - t V^T) at integer t points exactly.
    """
    V = np.array(V_input, dtype=object)
    VT = V.T
    vals = []
    for t in t_points:
        t = int(t)
        M = V - t * VT
        vals.append(det_bareiss_int(M))
    return vals

def interpolate_poly_from_samples(t_points, y_points):
    """
    Exact interpolation over QQ from (t, det) samples.
    """
    t = sp.Symbol("t")
    poly_expr = sp.interpolate(list(zip(t_points, y_points)), t)
    P = sp.Poly(sp.expand(poly_expr), t, domain=sp.QQ)
    return P

def poly_to_int_coeffs(P):
    """
    Convert a sympy Poly over QQ to an integer-coefficient Poly by clearing denominators.
    """
    t = P.gens[0]
    expr = P.as_expr()
    num, den = sp.fraction(expr)
    den = int(den)
    Pz = sp.Poly(num, t, domain=sp.ZZ)
    return Pz, den

def polyZZ_to_dict(Pz):
    """
    sympy Poly over ZZ -> dict exp->coeff (exp >= 0)
    """
    d = {}
    for mon, c in Pz.as_dict().items():
        # mon is a tuple like (exp,)
        e = int(mon[0])
        d[e] = int(c)
    if not d:
        d = {0: 0}
    return d

def normalise_alexander_from_power_poly(d_power):
    exps = sorted(d_power.keys())
    lo, hi = exps[0], exps[-1]
    # shift so (lo+hi)/2 becomes 0 (rounding to int shift)
    shift = -((lo + hi) // 2)
    d_sym = {e + shift: c for e, c in d_power.items()}

    # Fix overall sign to make Δ(1) positive if ±1; for knots Δ(1)=1.
    val1 = sum(d_sym.values())
    if val1 == -1:
        d_sym = {e: -c for e, c in d_sym.items()}
    return d_sym

def alexander_to_conway(alex_dict):
    """
    Convert symmetric Alexander dict exp->coeff to Conway dict deg->coeff (even degrees).
    """
    if not alex_dict:
        return {0: 1}

    max_k = max(abs(e) for e in alex_dict.keys())
    a0 = alex_dict.get(0, 0)

    # coefficients for (t^k + t^{-k})
    a = {}
    for k in range(1, max_k + 1):
        a[k] = 0.5 * (alex_dict.get(k, 0) + alex_dict.get(-k, 0))

    # Work in u = z^2. We need polynomials for (t^k + t^{-k}) as polynomials in u.
    # Recurrence for S_k(u) := t^k + t^{-k} with u = (t^{1/2}-t^{-1/2})^2 = t + t^{-1} - 2.
    # Then t + t^{-1} = u + 2, and
    # S_0 = 2, S_1 = u + 2, S_k = (u+2) S_{k-1} - S_{k-2}.
    def mul_uplus2(p):
        # (u+2)*p
        shifted = [0.0] + list(p)
        padded  = list(p) + [0.0]
        return [s + 2*pp for s, pp in zip(shifted, padded)]

    def sub(p1, p2):
        n = max(len(p1), len(p2))
        r = [0.0]*n
        for i,v in enumerate(p1): r[i] += v
        for i,v in enumerate(p2): r[i] -= v
        return r

    S = {0: [2.0], 1: [2.0, 1.0]}  # 2, (u+2)
    for k in range(2, max_k + 1):
        S[k] = sub(mul_uplus2(S[k-1]), S[k-2])

    # Build Conway in u
    conway_u = [0.0] * (max_k + 1)
    conway_u[0] += a0
    for k in range(1, max_k + 1):
        if abs(a.get(k, 0.0)) < 1e-12:
            continue
        for i, c in enumerate(S[k]):
            conway_u[i] += a[k] * c

    # Round to ints; map u^j -> z^(2j)
    out = {}
    for j, c in enumerate(conway_u):
        rc = int(round(c))
        if rc != 0:
            out[2*j] = rc
    return out if out else {0: 0}

def conway_from_seifert_interpolated(V_input, sample_shift=0):
    V = np.array(V_input, dtype=object)
    if V.size == 0:
        return {0: 1}

    n = V.shape[0]
    deg = n  # deg(det(V - t V^T)) <= n
    # Choose deg+1 distinct integer points, start at sample_shift
    t_points = list(range(sample_shift, sample_shift + deg + 1))

    y_points = det_values_seifert(V, t_points)
    P = interpolate_poly_from_samples(t_points, y_points)
    Pz, den = poly_to_int_coeffs(P)

    if den != 1:
        pass

    d_power = polyZZ_to_dict(Pz)
    alex = normalise_alexander_from_power_poly(d_power)
    conw = alexander_to_conway(alex)
    return conw

def conway_from_pd_snappy(pd_list):
    import snappy
    L = snappy.Link(pd_list)
    V = L.seifert_matrix()
    return conway_from_seifert_interpolated(V, sample_shift=0)