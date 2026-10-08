#!/usr/bin/env python3
"""
tau_kappa_sweep_selfcontained.py — tau_kappa sensitivity sweep on the 8-qubit
LPTE per-pair simulator benchmark (FakeFez, quietest path).

Self-contained by design: every helper below is copied VERBATIM from
FINAL_PYTHON/fez_preservation_test.py (the current results pipeline), so this
script reproduces exactly the pipeline's Stage-1 optimization, fidelity
evaluation (exact_dm_fidelity), and gate counting (count_2q_gates). The ONLY
new logic is an optional `pairs` filter threaded through the per-pair circuit
builders so that a pruned edge set (kappa(p) < tau_kappa) can be evaluated as
a reduced circuit.

Protocol (matches the paper's Stage 1-3 pruning protocol):
  Stage 1: optimize the full 7-edge / 21-term ansatz via run_dual_mma.
  Stage 2: for each threshold tau_kappa in {0.01, 0.02, 0.05, 0.1, 0.15, 0.2},
           register pruned edges  E_pruned = {p : kappa(p) < tau_kappa}.
  Stage 3: re-build the circuit on the surviving edges and evaluate DM fidelity
           (FakeFez exact_dm_fidelity), noiseless ideal fidelity, and 2Q gate
           count / depth after transpilation (count_2q_gates).
Output: a table printed to stdout and a JSON saved to
        FINAL_PYTHON/results/tau_kappa_sweep_<timestamp>.json

Run:  python3 FINAL_PYTHON/tau_kappa_sweep_selfcontained.py
      optional: --seeds 0,3 --iters 30 --thresholds 0.01,0.05,0.1,0.15
"""

import os
import time
import json
import warnings
import numpy as np
import scipy.linalg as la
import networkx as nx
from datetime import datetime

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeFez

warnings.filterwarnings("ignore")

# ==============================================================================
# VERBATIM FROM fez_preservation_test.py -- CONSTANTS & PAULI ALGEBRA
# ==============================================================================
N_QUBITS = 8
DIM = 2 ** N_QUBITS

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)

_PAIRS_2Q = {
    'XX': np.kron(_X, _X), 'XY': np.kron(_X, _Y), 'XZ': np.kron(_X, _Z),
    'YX': np.kron(_Y, _X), 'YY': np.kron(_Y, _Y), 'YZ': np.kron(_Y, _Z),
    'ZX': np.kron(_Z, _X), 'ZY': np.kron(_Z, _Y), 'ZZ': np.kron(_Z, _Z),
}
_NN = ['XX', 'YY', 'ZZ']
N_PAIRS = N_QUBITS - 1  # 7
N_TERMS = N_PAIRS * 3   # 21
N_PARAMS = 2 * N_TERMS  # 42

_TERM_INFO = [
    {'P2q': _PAIRS_2Q[pn], 'qk_qa': N_QUBITS - 2 - pi, 'qk_qb': N_QUBITS - 1 - pi}
    for pi in range(N_PAIRS) for pn in _NN
]

def _pauli_8q(p, qa, qb, n=8):
    p_mat = {'X': _X, 'Y': _Y, 'Z': _Z}[p]
    ops = [_I2] * n
    ops[qa] = p_mat
    ops[qb] = p_mat
    res = ops[-1]
    for op in ops[-2::-1]:
        res = np.kron(res, op)
    return res

_PAULI_8Q = []
for pi in range(N_PAIRS):
    qa = N_QUBITS - 2 - pi
    qb = N_QUBITS - 1 - pi
    for pn in _NN:
        _PAULI_8Q.append(_pauli_8q(pn[0], qa, qb))

_I_DIM = np.eye(DIM, dtype=complex)

def ts():
    return datetime.now().strftime("[%H:%M:%S]")

# ==============================================================================
# VERBATIM + `pairs` FILTER -- ANSATZ CIRCUIT BUILDER (per-pair / LPTE)
# ==============================================================================
def build_upte_local(params, pairs=None):
    """Per-pair (Local) PTE ansatz. VERBATIM from fez_preservation_test.py
    build_upte_local, plus optional `pairs` (sorted list of pair indices to
    keep; default None => all 7 pairs)."""
    rho_w = np.maximum(params[:N_TERMS], 1e-6)
    rho_n = rho_w / rho_w.sum()
    tau = params[N_TERMS:]
    qc = QuantumCircuit(N_QUBITS)
    for pi in range(N_PAIRS):
        if pairs is not None and pi not in pairs:
            continue
        qa = N_QUBITS - 2 - pi
        qb = N_QUBITS - 1 - pi
        B = np.zeros((4, 4), dtype=complex)
        for pp, pn in enumerate(_NN):
            j = pi * 3 + pp
            B += rho_n[j] * la.expm(-1j * tau[j] * _PAIRS_2Q[pn])
        Us, _, Vhs = la.svd(B)
        qc.unitary(Us @ Vhs, [qa, qb])
    return qc

# ==============================================================================
# VERBATIM -- TARGET TROTTER STATE & INVERSE
# ==============================================================================
def build_target(seed=0):
    rng = np.random.default_rng(seed)
    coeffs = rng.uniform(-0.5, 0.5, N_TERMS)
    qc = QuantumCircuit(N_QUBITS)
    k = 0
    for pi in range(N_PAIRS):
        for pn in _NN:
            qa = N_QUBITS - 2 - pi
            qb = N_QUBITS - 1 - pi
            qc.unitary(la.expm(-1j * coeffs[k] * _PAIRS_2Q[pn]), [qa, qb])
            k += 1
    sv_sim = AerSimulator(method='statevector')
    qc_sv = qc.copy()
    qc_sv.save_statevector()
    sv = sv_sim.run(transpile(qc_sv, sv_sim, optimization_level=0), shots=1).result().data(0)['statevector']
    return np.array(sv), coeffs, qc

def build_target_inv_circuit(coeffs):
    layers = []
    k = 0
    for pi in range(N_PAIRS):
        for pn in _NN:
            qa = N_QUBITS - 2 - pi
            qb = N_QUBITS - 1 - pi
            layers.append((coeffs[k], _PAIRS_2Q[pn], qa, qb))
            k += 1
    qc = QuantumCircuit(N_QUBITS)
    for c, P2q, qa, qb in reversed(layers):
        qc.unitary(la.expm(+1j * c * P2q), [qa, qb])
    return qc

def init_params(seed=42):
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.01, 1.0, N_TERMS)
    tau = rng.uniform(-np.pi, np.pi, N_TERMS)
    return np.concatenate([rho, tau])

# ==============================================================================
# VERBATIM -- TOPOLOGY & SIMULATOR SETUP
# ==============================================================================
def _get_coupling_map(backend):
    cm = getattr(backend, 'coupling_map', None)
    if cm is not None:
        raw_edges = list(cm) if not callable(cm) else list(cm())
        return CouplingMap(raw_edges) if not isinstance(cm, CouplingMap) else cm
    cfg_cm = backend.configuration().coupling_map
    return CouplingMap(cfg_cm) if cfg_cm is not None else None

def _get_basis_gates(backend):
    bg = getattr(backend, 'basis_gates', None)
    if bg is not None:
        return list(bg) if not callable(bg) else list(bg())
    return list(backend.configuration().basis_gates)

def find_quietest_path(backend):
    props = backend.properties() if hasattr(backend, 'properties') else None
    bg = _get_basis_gates(backend)
    tq = [g for g in bg if g in ['cx', 'cz', 'ecr']][0]
    cm = _get_coupling_map(backend)
    G = nx.Graph()
    for q1, q2 in cm.get_edges():
        e1 = float(props.gate_error(tq, [q1, q2]) or 0.0) if props and hasattr(props, 'gate_error') else 0.0
        e2 = float(props.gate_error(tq, [q2, q1]) or 0.0) if props and hasattr(props, 'gate_error') else 0.0
        r1 = float(props.readout_error(q1) or 0.0) if props and hasattr(props, 'readout_error') else 0.0
        r2 = float(props.readout_error(q2) or 0.0) if props and hasattr(props, 'readout_error') else 0.0
        G.add_edge(q1, q2, weight=(e1 + e2) / 2 + 0.1 * (r1 + r2) / 2)
    paths = []
    def dfs(n, p):
        if len(p) == 8:
            paths.append(p)
            return
        for nb in G.neighbors(n):
            if nb not in p:
                dfs(nb, p + [nb])

    for n in G.nodes:
        dfs(n, [n])
    scored = sorted([(sum(G[p[i]][p[i + 1]]['weight'] for i in range(7)), p) for p in paths])
    print(f"{ts()} Top quietest path on {getattr(backend, 'name', 'backend')}: {scored[0][1]} (Score: {scored[0][0]:.5f})")
    return scored[0][1]

_quietest_fake_cache = None

def get_quietest_fake_path():
    global _quietest_fake_cache
    if _quietest_fake_cache is None:
        _quietest_fake_cache = find_quietest_path(FakeFez())
    return _quietest_fake_cache

def build_reduced_sim(backend, layout_8):
    p2v = {p: i for i, p in enumerate(layout_8)}
    fc = _get_coupling_map(backend).get_edges()
    red_edges = [(p2v[a], p2v[b]) for a, b in fc if a in layout_8 and b in layout_8]
    cmap = CouplingMap(red_edges)
    bg = _get_basis_gates(backend)
    props = backend.properties()
    nm = NoiseModel(basis_gates=bg)
    for phys_q in layout_8:
        vq = p2v[phys_q]
        ro_err = props.readout_error(phys_q) if props else 0.0
        if ro_err > 0:
            p0g0 = max(1.0 - ro_err, 1e-4)
            nm.add_readout_error([[p0g0, 1 - p0g0], [1 - p0g0, p0g0]], [vq])
    for (p1, p2) in fc:
        if p1 not in layout_8 or p2 not in layout_8:
            continue
        v1, v2 = p2v[p1], p2v[p2]
        for gn in bg:
            if gn not in ("cx", "cz", "ecr"):
                continue
            ge = props.gate_error(gn, [p1, p2]) if props else 0.0
            if ge and ge > 0:
                nm.add_quantum_error(depolarizing_error(ge, 2), gn, [v1, v2])
    if not nm._local_quantum_errors:
        raise RuntimeError("noise model has no quantum errors after build -- gate noise "
                           "was silently dropped and must not be hidden")
    sim_dm = AerSimulator(noise_model=nm, coupling_map=cmap, method='density_matrix')
    sim_meas = AerSimulator(noise_model=nm, coupling_map=cmap)
    return sim_dm, sim_meas, cmap

def get_default_sims():
    path = get_quietest_fake_path()
    return build_reduced_sim(FakeFez(), path)

# ==============================================================================
# VERBATIM + `pairs` FILTER -- LOSS FUNCTIONS & DENSITY MATRIX OBJECTIVES
# ==============================================================================
def exact_dm_fidelity(qc_ansatz, psi_target, dm_sim, basis_gates=None, coupling_map=None):
    fb = FakeFez()
    if basis_gates is None:
        basis_gates = _get_basis_gates(fb)
    cm = _get_coupling_map(dm_sim) if coupling_map is None else coupling_map
    qc_t = transpile(qc_ansatz, coupling_map=cm, basis_gates=basis_gates,
                     initial_layout=list(range(N_QUBITS)), optimization_level=3)
    qc_t.save_density_matrix()
    rho = np.array(dm_sim.run(qc_t, shots=1).result().data(0)['density_matrix'])
    return float(np.clip(np.real(psi_target.conj() @ rho @ psi_target), 0, 1))

def compute_ideal_state(params, pairs=None):
    rho_w = np.maximum(params[:N_TERMS], 1e-6)
    rho_n = rho_w / rho_w.sum()
    tau = params[N_TERMS:]
    psi = np.zeros((2,) * N_QUBITS, dtype=complex)
    psi[(0,) * N_QUBITS] = 1.0

    def _apply_gate(U, psi, qi, qj):
        N = psi.ndim
        ax = [qi, qj] + [k for k in range(N) if k not in (qi, qj)]
        psi = np.transpose(psi, ax).reshape(4, -1)
        psi = (U @ psi).reshape((2, 2) + (2,) * (N - 2))
        return np.transpose(psi, np.argsort(ax))

    for pi in range(N_PAIRS):
        if pairs is not None and pi not in pairs:
            continue
        qa = N_QUBITS - 2 - pi
        qb = N_QUBITS - 1 - pi
        B = np.zeros((4, 4), dtype=complex)
        for pp, pn in enumerate(_NN):
            j = pi * 3 + pp
            B += rho_n[j] * la.expm(-1j * tau[j] * _PAIRS_2Q[pn])
        Us, _, Vhs = la.svd(B)
        psi = _apply_gate(Us @ Vhs, psi, qa, qb)

    return np.transpose(psi, range(N_QUBITS - 1, -1, -1)).flatten()

def _reg_penalty(params, pairs=None):
    rho_w = np.maximum(params[:N_TERMS], 1e-6)
    rho_n = rho_w / rho_w.sum()
    tau = params[N_TERMS:]
    total = 0.0
    for pi in range(N_PAIRS):
        if pairs is not None and pi not in pairs:
            continue
        B = np.zeros((4, 4), dtype=complex)
        for pp, pn in enumerate(_NN):
            j = pi * 3 + pp
            B += rho_n[j] * la.expm(-1j * tau[j] * _PAIRS_2Q[pn])
        dev = B.conj().T @ B - np.eye(4, dtype=complex)
        total += np.real(np.trace(dev.conj().T @ dev))
    return total

def _noisy_dm(params, simulator, pairs=None):
    rho_w = np.maximum(params[:N_TERMS], 1e-6)
    rho_n = rho_w / rho_w.sum()
    tau = params[N_TERMS:]
    qc = QuantumCircuit(N_QUBITS)

    for pi in range(N_PAIRS):
        if pairs is not None and pi not in pairs:
            continue
        qa = N_QUBITS - 2 - pi
        qb = N_QUBITS - 1 - pi
        B = np.zeros((4, 4), dtype=complex)
        for pp, pn in enumerate(_NN):
            j = pi * 3 + pp
            B += rho_n[j] * la.expm(-1j * tau[j] * _PAIRS_2Q[pn])
        Us, _, Vhs = la.svd(B)
        qc.unitary(Us @ Vhs, [qa, qb])

    qc_t = transpile(qc, coupling_map=_get_coupling_map(simulator),
                     basis_gates=_get_basis_gates(FakeFez()),
                     initial_layout=list(range(N_QUBITS)), optimization_level=3)
    qc_t.save_density_matrix()
    return np.array(simulator.run(qc_t, shots=1).result().data(0)['density_matrix'])

def noise_aware_objective(params, psi_target, simulator, pairs=None):
    psi_ideal = compute_ideal_state(params, pairs=pairs)
    rho_noisy = _noisy_dm(params, simulator, pairs=pairs)
    ideal_fid = float(np.abs(np.vdot(psi_target, psi_ideal)) ** 2)
    noisy_fid = float(np.clip(np.real(psi_target.conj() @ rho_noisy @ psi_target), 0, 1))
    loss = (1.0 - noisy_fid) + 0.1 * (1.0 - ideal_fid) + 1e-3 * _reg_penalty(params, pairs=pairs)
    return loss, noisy_fid, ideal_fid

# ==============================================================================
# VERBATIM -- DUAL-MMA OPTIMIZER
# ==============================================================================
class MMAOptimizer:
    def __init__(self, n, move_limit=0.4, gamma=0.5):
        self.n = n
        self.move_limit = move_limit
        self.gamma = gamma
        self.L = self.U = self.prev_loss = None

    def init(self, p0, delta=0.6):
        self.L = p0 - delta
        self.U = p0 + delta

    def step(self, x, grad):
        x_new = np.zeros_like(x)
        for i in range(self.n):
            g, xi, Li, Ui = grad[i], x[i], self.L[i], self.U[i]
            pi = abs(g) * (Ui - xi) ** 2 if g < 0 else 0.0
            qi = abs(g) * (xi - Li) ** 2 if g >= 0 else 0.0
            dn = pi / (Ui - xi + 1e-12) ** 2 + qi / (xi - Li + 1e-12) ** 2
            x_new[i] = (xi + (pi / (Ui - xi + 1e-12) - qi / (xi - Li + 1e-12)) / dn if dn > 1e-12 else xi)
            x_new[i] = np.clip(x_new[i], max(xi - self.move_limit, Li + 1e-6), min(xi + self.move_limit, Ui - 1e-6))
        return x_new

    def update(self, x, loss):
        good = self.prev_loss is None or loss < self.prev_loss - 1e-8
        s = 1.2 / self.gamma if good else self.gamma
        self.L = x - s * (x - self.L)
        self.U = x + s * (self.U - x)
        self.L = np.minimum(self.L, self.U - 1e-4)
        self.U = np.maximum(self.U, self.L + 1e-4)
        self.prev_loss = loss

def run_dual_mma(init_params, psi_target, simulator, max_iters=40, pairs=None):
    _prev_grad = [None]

    def _hybrid_gradient(params):
        grad = np.zeros_like(params)
        for i in range(N_TERMS):
            pp = np.clip(params.copy(), 1e-6, None); pp[i] += 1e-4
            pm = np.clip(params.copy(), 1e-6, None); pm[i] -= 1e-4
            grad[i] = (noise_aware_objective(pp, psi_target, simulator, pairs=pairs)[0] -
                       noise_aware_objective(pm, psi_target, simulator, pairs=pairs)[0]) / 2e-4
        for i in range(N_TERMS, N_PARAMS):
            pp = params.copy(); pp[i] += np.pi / 4
            pm = params.copy(); pm[i] -= np.pi / 4
            grad[i] = (noise_aware_objective(pp, psi_target, simulator, pairs=pairs)[0] -
                       noise_aware_objective(pm, psi_target, simulator, pairs=pairs)[0]) / (np.pi / 2)
        if _prev_grad[0] is None:
            _prev_grad[0] = grad.copy()
            return grad
        g = 0.4 * grad + 0.6 * _prev_grad[0]
        _prev_grad[0] = g.copy()
        return g

    mma_r = MMAOptimizer(N_TERMS, move_limit=0.2)
    mma_r.init(init_params[:N_TERMS], delta=0.4)
    mma_t = MMAOptimizer(N_TERMS, move_limit=0.6)
    mma_t.init(init_params[N_TERMS:], delta=0.8)

    cur = init_params.copy()
    cur_loss, cur_fid, cur_ifid = noise_aware_objective(cur, psi_target, simulator, pairs=pairs)
    best_fid, best_params, stag = cur_fid, cur.copy(), 0
    print(f"{ts()} Init DM: {cur_fid:.4f} (ideal {cur_ifid:.4f})")

    for it in range(max_iters):
        t0 = time.time()
        grad = _hybrid_gradient(cur)
        new = np.concatenate([mma_r.step(cur[:N_TERMS], grad[:N_TERMS]),
                              mma_t.step(cur[N_TERMS:], grad[N_TERMS:])])
        new_loss, new_fid, new_ifid = noise_aware_objective(new, psi_target, simulator, pairs=pairs)
        delta = new_fid - cur_fid

        if new_loss < cur_loss - 1e-6 or delta > -1e-5:
            cur, cur_loss, cur_fid, cur_ifid = new, new_loss, new_fid, new_ifid
            if cur_fid > best_fid:
                best_fid, best_params, stag = cur_fid, cur.copy(), 0
            else:
                stag += 1
            mma_r.move_limit = min(0.4, mma_r.move_limit * (1.4 if delta > 0.01 else 1.2 if delta > 0.001 else 0.9))
        else:
            mma_r.move_limit = max(0.02, mma_r.move_limit * 0.8)
            stag += 1

        mma_t.move_limit = mma_r.move_limit * 2.0
        mma_r.update(cur[:N_TERMS], cur_loss)
        mma_t.update(cur[N_TERMS:], cur_loss)
        print(f"{ts()} Iter {it + 1:2d}: DM={cur_fid:.4f} (ideal {cur_ifid:.4f}) Δ={delta:+.4f} | {time.time() - t0:.1f}s | stag={stag}")
        if stag > 15:
            print(f"{ts()} Stagnated at iter {it + 1}.")
            break

    print(f"{ts()} Optimisation complete. Best DM fidelity: {best_fid:.4f}")
    return best_params

# ==============================================================================
# VERBATIM -- GATE COUNTING
# ==============================================================================
def count_2q_gates(qc, coupling_map=None, basis_gates=None):
    if basis_gates is None:
        basis_gates = _get_basis_gates(FakeFez())
    cm = CouplingMap.from_line(N_QUBITS) if coupling_map is None else coupling_map
    qc_t = transpile(qc, coupling_map=cm, basis_gates=basis_gates,
                     initial_layout=list(range(N_QUBITS)), optimization_level=3)
    ops = qc_t.count_ops()
    n2q = sum(v for k, v in ops.items() if k in ('cx', 'cy', 'cz', 'ecr', 'swap'))
    return n2q, qc_t.depth()

# ==============================================================================
# NEW DRIVER (thin): kappa(p) computation + tau_kappa sweep table
# ==============================================================================
def pair_kappas(params):
    """Per-pair densities kappa(p) = sum_P r_{p,P} under the global softmax
    used by build_upte_local (rho_w = max(params,1e-6); rho_n = w/sum w)."""
    rho_w = np.maximum(params[:N_TERMS], 1e-6)
    rho_n = rho_w / rho_w.sum()
    return [float(rho_n[pi * 3:(pi + 1) * 3].sum()) for pi in range(N_PAIRS)]

def sweep_seed(seed, sim_dm, sim_meas, cmap, bg, psi_target, threshold_list, max_iters):
    p0 = init_params(seed=seed)
    opt_params = run_dual_mma(p0, psi_target, sim_dm, max_iters=max_iters, pairs=None)

    kappas = pair_kappas(opt_params)
    full_qc = build_upte_local(opt_params, pairs=None)
    full_dm = exact_dm_fidelity(full_qc, psi_target, sim_dm, bg, cmap)
    full_ideal = float(np.abs(np.vdot(psi_target, compute_ideal_state(opt_params, pairs=None))) ** 2)
    full_2q, full_depth = count_2q_gates(full_qc, cmap, bg)

    rows = [{
        "seed": seed, "tau_kappa": 0.0, "n_edges": N_PAIRS,
        "pruned": [], "kappa": kappas,
        "dm_fidelity": round(full_dm, 4), "ideal_fidelity": round(full_ideal, 4),
        "n_2q": int(full_2q), "depth": int(full_depth),
    }]

    for tk in threshold_list:
        pruned = [pi for pi in range(N_PAIRS) if kappas[pi] < tk]
        survivors = [pi for pi in range(N_PAIRS) if kappas[pi] >= tk]
        qc = build_upte_local(opt_params, pairs=survivors)
        dm = exact_dm_fidelity(qc, psi_target, sim_dm, bg, cmap)
        ideal = float(np.abs(np.vdot(psi_target, compute_ideal_state(opt_params, pairs=survivors))) ** 2)
        n2q, depth = count_2q_gates(qc, cmap, bg)
        rows.append({
            "seed": seed, "tau_kappa": tk, "n_edges": len(survivors),
            "pruned": pruned, "kappa": kappas,
            "dm_fidelity": round(dm, 4), "ideal_fidelity": round(ideal, 4),
            "n_2q": int(n2q), "depth": int(depth),
        })
    return rows

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,3", help="comma-separated seeds")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--thresholds", default="0.01,0.05,0.1,0.15,0.2",
                        help="comma-separated tau_kappa thresholds")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    threshold_list = [float(t) for t in args.thresholds.split(",")]

    sim_dm, sim_meas, cmap = get_default_sims()
    fb = FakeFez()
    bg = _get_basis_gates(fb)

    psi_target, coeffs, target_qc = build_target(seed=0)
    print(f"{ts()} Target built (seed 0).")

    all_rows = []
    for seed in seeds:
        print(f"\n{'=' * 60}\n{ts()} Seed {seed} | Stage-1 full optimization\n{'=' * 60}")
        rows = sweep_seed(seed, sim_dm, sim_meas, cmap, bg, psi_target, threshold_list, args.iters)
        all_rows.extend(rows)

    print("\n\n" + "=" * 88)
    print("TAU_KAPPA SENSITIVITY SWEEP -- 8q FakeFez LPTE per-pair")
    print("=" * 88)
    header = f"{'seed':>4} {'tau_k':>5} {'#edges':>6} {'DM_fid':>8} {'ideal':>7} {'2Q':>4} {'depth':>5}  pruned"
    print(header)
    print("-" * 88)
    for r in all_rows:
        print(f"{r['seed']:>4} {r['tau_kappa']:>5.2f} {r['n_edges']:>6} {r['dm_fidelity']:>8.4f} "
              f"{r['ideal_fidelity']:>7.4f} {r['n_2q']:>4} {r['depth']:>5}  {r['pruned']}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"tau_kappa_sweep_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out_path, "w") as f:
        json.dump({"seed": seeds[0], "thresholds": threshold_list, "rows": all_rows}, f, indent=2)
    print(f"\n{ts()} Saved sweep to {out_path}")

if __name__ == "__main__":
    main()
