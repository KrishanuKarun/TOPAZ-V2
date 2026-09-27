#!/usr/bin/env python3
"""
Fez 8-qubit state-preservation test (Loschmidt echo) — Cleaned & Refactored.
Configured for LPTE (Local/Per-Pair PTE) execution.
"""

import os
import sys
import time
import json
import warnings
import argparse
import numpy as np
import scipy.linalg as la
import networkx as nx
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeFez

warnings.filterwarnings("ignore")

# ==============================================================================
# CONSTANTS & PAULI ALGEBRA
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
# ANSATZ CIRCUIT BUILDERS
# ==============================================================================
def build_circuit(params):
    """Product-PTE ansatz."""
    rho_w = np.maximum(params[:N_TERMS], 1e-6)
    rho_n = rho_w / rho_w.sum()
    tau = params[N_TERMS:]
    qc = QuantumCircuit(N_QUBITS)
    for j, t in enumerate(_TERM_INFO):
        qc.unitary(la.expm(-1j * rho_n[j] * tau[j] * t['P2q']), [t['qk_qa'], t['qk_qb']])
    return qc

def build_upte_local(params):
    """Per-pair (Local) PTE ansatz."""
    rho_w = np.maximum(params[:N_TERMS], 1e-6)
    rho_n = rho_w / rho_w.sum()
    tau = params[N_TERMS:]
    qc = QuantumCircuit(N_QUBITS)
    for pi in range(N_PAIRS):
        qa = N_QUBITS - 2 - pi
        qb = N_QUBITS - 1 - pi
        B = np.zeros((4, 4), dtype=complex)
        for pp, pn in enumerate(_NN):
            j = pi * 3 + pp
            B += rho_n[j] * la.expm(-1j * tau[j] * _PAIRS_2Q[pn])
        Us, _, Vhs = la.svd(B)
        qc.unitary(Us @ Vhs, [qa, qb])
    return qc

def build_upte_global(params):
    """Global-PTE ansatz."""
    rho_w = np.maximum(params[:N_TERMS], 1e-6)
    rho_n = rho_w / rho_w.sum()
    tau = params[N_TERMS:]
    B = np.zeros((DIM, DIM), dtype=complex)
    for j in range(N_TERMS):
        B += rho_n[j] * (np.cos(tau[j]) * _I_DIM - 1j * np.sin(tau[j]) * _PAULI_8Q[j])
    Us, _, Vhs = la.svd(B)
    qc = QuantumCircuit(N_QUBITS)
    qc.unitary(Us @ Vhs, range(N_QUBITS))
    return qc

def build_ansatz(params, mode="per_pair"):
    if mode == "product":
        return build_circuit(params)
    if mode == "per_pair":
        return build_upte_local(params)
    if mode == "global":
        return build_upte_global(params)
    raise ValueError(f"Unknown mode: {mode}")

def _use_upte_flag(mode):
    if mode == "per_pair":
        return True
    if mode == "global":
        return "global"
    return False

# ==============================================================================
# TARGET TROTTER STATE & INVERSE
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

def baseline_params(seed=0):
    """Original V3 untuned product baseline: random params, no MMA."""
    rng = np.random.default_rng(seed + 42)
    rho = rng.uniform(0.1, 0.4, N_TERMS)
    tau = rng.uniform(0.1, np.pi, N_TERMS)
    return np.concatenate([rho, tau])

# ==============================================================================
# TOPOLOGY & SIMULATOR SETUP
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
# LOSS FUNCTIONS & DENSITY MATRIX OBJECTIVES
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

def compute_ideal_state(params, use_upte=False):
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

    if use_upte == 'global':
        B = np.zeros((DIM, DIM), dtype=complex)
        for j in range(N_TERMS):
            B += rho_n[j] * (np.cos(tau[j]) * _I_DIM - 1j * np.sin(tau[j]) * _PAULI_8Q[j])
        Us, _, Vhs = la.svd(B)
        psi = (Us @ Vhs @ psi.flatten()).reshape((2,) * N_QUBITS)
    elif use_upte:
        for pi in range(N_PAIRS):
            qa = N_QUBITS - 2 - pi
            qb = N_QUBITS - 1 - pi
            B = np.zeros((4, 4), dtype=complex)
            for pp, pn in enumerate(_NN):
                j = pi * 3 + pp
                B += rho_n[j] * la.expm(-1j * tau[j] * _PAIRS_2Q[pn])
            Us, _, Vhs = la.svd(B)
            psi = _apply_gate(Us @ Vhs, psi, qa, qb)
    else:
        for j, t in enumerate(_TERM_INFO):
            psi = _apply_gate(la.expm(-1j * rho_n[j] * tau[j] * t['P2q']), psi, t['qk_qa'], t['qk_qb'])

    return np.transpose(psi, range(N_QUBITS - 1, -1, -1)).flatten()

def _reg_penalty(params, use_upte=False):
    rho_w = np.maximum(params[:N_TERMS], 1e-6)
    rho_n = rho_w / rho_w.sum()
    tau = params[N_TERMS:]
    if use_upte == 'global':
        B = np.zeros((DIM, DIM), dtype=complex)
        for j in range(N_TERMS):
            B += rho_n[j] * (np.cos(tau[j]) * _I_DIM - 1j * np.sin(tau[j]) * _PAULI_8Q[j])
        _, S, _ = la.svd(B)
        return float(np.sum((S ** 2 - 1.0) ** 2))
    total = 0.0
    for pi in range(N_PAIRS):
        B = np.zeros((4, 4), dtype=complex)
        for pp, pn in enumerate(_NN):
            j = pi * 3 + pp
            B += rho_n[j] * la.expm(-1j * tau[j] * _PAIRS_2Q[pn])
        dev = B.conj().T @ B - np.eye(4, dtype=complex)
        total += np.real(np.trace(dev.conj().T @ dev))
    return total

def _noisy_dm(params, simulator, use_upte=False):
    rho_w = np.maximum(params[:N_TERMS], 1e-6)
    rho_n = rho_w / rho_w.sum()
    tau = params[N_TERMS:]
    qc = QuantumCircuit(N_QUBITS)
    skip_transpile = False

    if use_upte == 'global':
        B = np.zeros((DIM, DIM), dtype=complex)
        for j in range(N_TERMS):
            B += rho_n[j] * (np.cos(tau[j]) * _I_DIM - 1j * np.sin(tau[j]) * _PAULI_8Q[j])
        Us, _, Vhs = la.svd(B)
        qc.unitary(Us @ Vhs, range(N_QUBITS))
        skip_transpile = True
    elif use_upte:
        for pi in range(N_PAIRS):
            qa = N_QUBITS - 2 - pi
            qb = N_QUBITS - 1 - pi
            B = np.zeros((4, 4), dtype=complex)
            for pp, pn in enumerate(_NN):
                j = pi * 3 + pp
                B += rho_n[j] * la.expm(-1j * tau[j] * _PAIRS_2Q[pn])
            Us, _, Vhs = la.svd(B)
            qc.unitary(Us @ Vhs, [qa, qb])
    else:
        for j, t in enumerate(_TERM_INFO):
            qc.unitary(la.expm(-1j * rho_n[j] * tau[j] * t['P2q']), [t['qk_qa'], t['qk_qb']])

    if skip_transpile:
        qc.save_density_matrix()
        return np.array(simulator.run(qc, shots=1).result().data(0)['density_matrix'])

    qc_t = transpile(qc, coupling_map=_get_coupling_map(simulator),
                     basis_gates=_get_basis_gates(FakeFez()),
                     initial_layout=list(range(N_QUBITS)), optimization_level=3)
    qc_t.save_density_matrix()
    return np.array(simulator.run(qc_t, shots=1).result().data(0)['density_matrix'])

def noise_aware_objective(params, psi_target, simulator, use_upte=False):
    psi_ideal = compute_ideal_state(params, use_upte=use_upte)
    rho_noisy = _noisy_dm(params, simulator, use_upte=use_upte)
    ideal_fid = float(np.abs(np.vdot(psi_target, psi_ideal)) ** 2)
    noisy_fid = float(np.clip(np.real(psi_target.conj() @ rho_noisy @ psi_target), 0, 1))
    loss = (1.0 - noisy_fid) + 0.1 * (1.0 - ideal_fid) + 1e-3 * _reg_penalty(params, use_upte=use_upte)
    return loss, noisy_fid, ideal_fid

# ==============================================================================
# DUAL-MMA OPTIMIZER
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

def run_dual_mma(init_params, psi_target, simulator, use_upte=False, max_iters=40):
    _prev_grad = [None]

    def _hybrid_gradient(params):
        grad = np.zeros_like(params)
        for i in range(N_TERMS):
            pp = np.clip(params.copy(), 1e-6, None); pp[i] += 1e-4
            pm = np.clip(params.copy(), 1e-6, None); pm[i] -= 1e-4
            grad[i] = (noise_aware_objective(pp, psi_target, simulator, use_upte=use_upte)[0] -
                       noise_aware_objective(pm, psi_target, simulator, use_upte=use_upte)[0]) / 2e-4
        for i in range(N_TERMS, N_PARAMS):
            pp = params.copy(); pp[i] += np.pi / 4
            pm = params.copy(); pm[i] -= np.pi / 4
            grad[i] = (noise_aware_objective(pp, psi_target, simulator, use_upte=use_upte)[0] -
                       noise_aware_objective(pm, psi_target, simulator, use_upte=use_upte)[0]) / (np.pi / 2)
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

    mode_lbl = "per-pair" if use_upte is True else ("global" if use_upte == 'global' else "product")
    cur = init_params.copy()
    cur_loss, cur_fid, cur_ifid = noise_aware_objective(cur, psi_target, simulator, use_upte=use_upte)
    best_fid, best_params, stag = cur_fid, cur.copy(), 0
    print(f"{ts()} Init DM ({mode_lbl}): {cur_fid:.4f} (ideal {cur_ifid:.4f})")

    for it in range(max_iters):
        t0 = time.time()
        grad = _hybrid_gradient(cur)
        new = np.concatenate([mma_r.step(cur[:N_TERMS], grad[:N_TERMS]),
                              mma_t.step(cur[N_TERMS:], grad[N_TERMS:])])
        new_loss, new_fid, new_ifid = noise_aware_objective(new, psi_target, simulator, use_upte=use_upte)
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
# LOSCHMIDT ECHO BENCHMARK & MITIGATION
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

def _prepend_probe(qc, probe):
    if probe in ("zero", "0"):
        return qc
    if probe == "plus":
        p = QuantumCircuit(N_QUBITS)
        p.h(range(N_QUBITS))
        return p.compose(qc)
    raise ValueError(f"Unknown probe: {probe}")

def dfe_fidelity(qc_ansatz, target_inv_circuit, meas_backend, shots=2000, label="",
                 physical_layout=None, probe="zero"):
    qc = qc_ansatz.compose(target_inv_circuit)
    qc = _prepend_probe(qc, probe)
    if probe == "plus":
        qc.h(range(N_QUBITS))
    qc.measure_all()
    ret_state = '0' * N_QUBITS

    if physical_layout is not None:
        from qiskit_ibm_runtime import SamplerV2
        qc_t = transpile(qc, backend=meas_backend, initial_layout=physical_layout, optimization_level=3)
        sampler = SamplerV2(mode=meas_backend)
        print(f"{ts()} [{label}] DFE ({shots} shots, hardware layout={physical_layout})")
        job = sampler.run([(qc_t,)], shots=shots)
        counts = job.result()[0].data.meas.get_counts()
        fid = float(np.clip(counts.get(ret_state, 0) / shots, 0, 1))
        return fid

    basis_gates = _get_basis_gates(FakeFez())
    cm = _get_coupling_map(meas_backend)
    qc_t = transpile(qc, coupling_map=cm, basis_gates=basis_gates,
                     initial_layout=list(range(N_QUBITS)), optimization_level=3)
    chunk, total_zero, total_shots = 200, 0, 0
    while total_shots < shots:
        this = min(chunk, shots - total_shots)
        counts = meas_backend.run(qc_t, shots=this).result().get_counts()
        total_zero += counts.get(ret_state, 0)
        total_shots += this
    return float(np.clip(total_zero / total_shots, 0, 1))

def noiseless_echo_fidelity(qc_ansatz, target_inv_circuit, probe="zero"):
    qc = qc_ansatz.compose(target_inv_circuit)
    qc = _prepend_probe(qc, probe)
    if probe == "plus":
        qc.h(range(N_QUBITS))
    sv_sim = AerSimulator(method='statevector')
    qc_t = transpile(qc, sv_sim, optimization_level=0)
    qc_t.save_statevector()
    sv = np.array(sv_sim.run(qc_t, shots=1).result().data(0)["statevector"])
    return float(np.abs(sv[0]) ** 2)

def build_tpnm_calibration(meas_backend, cmap, basis_gates, shots=5000, physical_layout=None):
    from qiskit_ibm_runtime import SamplerV2

    def _run_cal(qc):
        qc.measure_all()
        if physical_layout is not None:
            qc_t = transpile(qc, backend=meas_backend, initial_layout=physical_layout, optimization_level=3)
            job = SamplerV2(mode=meas_backend).run([(qc_t,)], shots=shots)
            return job.result()[0].data.meas.get_counts()
        qc_t = transpile(qc, coupling_map=cmap, basis_gates=basis_gates,
                         initial_layout=list(range(N_QUBITS)), optimization_level=3)
        return meas_backend.run(qc_t, shots=shots).result().get_counts()

    qc0 = QuantumCircuit(N_QUBITS, N_QUBITS)
    cal0 = _run_cal(qc0)

    qc1 = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc1.x(range(N_QUBITS))
    cal1 = _run_cal(qc1)

    conf = np.zeros((N_QUBITS, 2, 2))
    for qi in range(N_QUBITS):
        n0g0 = sum(v for k, v in cal0.items() if k[N_QUBITS - 1 - qi] == "0")
        n1g1 = sum(v for k, v in cal1.items() if k[N_QUBITS - 1 - qi] == "1")
        p0g0 = max(n0g0 / shots, 1e-4)
        p1g1 = max(n1g1 / shots, 1e-4)
        conf[qi] = [[p0g0, 1 - p1g1], [1 - p0g0, p1g1]]

    tpnm_matrix = conf[N_QUBITS - 1]
    for qi in range(N_QUBITS - 2, -1, -1):
        tpnm_matrix = np.kron(tpnm_matrix, conf[qi])
    return tpnm_matrix

def inversion_test_mitigated(qc_ansatz, target_inv, meas_backend, cmap, basis_gates,
                            tpnm_matrix, shots, label, physical_layout=None, probe="zero"):
    qc = qc_ansatz.compose(target_inv)
    qc = _prepend_probe(qc, probe)
    if probe == "plus":
        qc.h(range(N_QUBITS))
    qc.measure_all()

    if physical_layout is not None:
        from qiskit_ibm_runtime import SamplerV2
        qc_t = transpile(qc, backend=meas_backend, initial_layout=physical_layout, optimization_level=3)
        job = SamplerV2(mode=meas_backend).run([(qc_t,)], shots=shots)
        counts = job.result()[0].data.meas.get_counts()
        obs = np.zeros(DIM)
        for bs, ct in counts.items():
            obs[int(bs, 2)] = ct
        true_counts = np.maximum(np.linalg.lstsq(tpnm_matrix, obs, rcond=None)[0], 0)
        return float(np.clip(true_counts[0] / shots, 0, 1))

    qc_t = transpile(qc, coupling_map=cmap, basis_gates=basis_gates,
                     initial_layout=list(range(N_QUBITS)), optimization_level=3)
    chunk, total_00, total = 200, 0, 0
    while total < shots:
        this = min(chunk, shots - total)
        counts = meas_backend.run(qc_t, shots=this).result().get_counts()
        obs = np.zeros(DIM)
        for bs, ct in counts.items():
            obs[int(bs, 2)] = ct
        true_counts = np.maximum(np.linalg.lstsq(tpnm_matrix, obs, rcond=None)[0], 0)
        total_00 += true_counts[0]
        total += this
    return float(np.clip(total_00 / total, 0, 1))

def fold_circuit(qc, noise_level):
    if noise_level == 1.0:
        return qc.copy()
    copies = (int(2 * noise_level - 1)) | 1
    folded = QuantumCircuit(*qc.qregs, *qc.cregs)
    for item in qc.data:
        instr = item.operation
        if instr.name in ('measure', 'barrier', 'reset'):
            folded.append(instr, item.qubits, item.clbits)
            continue
        for i in range(copies):
            op = instr.inverse() if i % 2 else instr
            folded.append(op, item.qubits, item.clbits)
    return folded

def zne_echo_fidelity(qc_ansatz, target_inv, meas_backend, shots, label,
                      scale_factors=(1, 2, 3), physical_layout=None, probe="zero"):
    fids = []
    for sf in scale_factors:
        qc = qc_ansatz.compose(target_inv)
        qc = _prepend_probe(qc, probe)
        if probe == "plus":
            qc.h(range(N_QUBITS))
        qc_f = fold_circuit(qc, sf)
        qc_f.measure_all()
        ret_state = '0' * N_QUBITS
        if physical_layout is not None:
            from qiskit_ibm_runtime import SamplerV2
            qc_t = transpile(qc_f, backend=meas_backend, initial_layout=physical_layout, optimization_level=3)
            counts = SamplerV2(mode=meas_backend).run([(qc_t,)], shots=shots).result()[0].data.meas.get_counts()
        else:
            basis_gates = _get_basis_gates(FakeFez())
            cm = _get_coupling_map(meas_backend)
            qc_t = transpile(qc_f, coupling_map=cm, basis_gates=basis_gates,
                             initial_layout=list(range(N_QUBITS)), optimization_level=3)
            counts = meas_backend.run(qc_t, shots=shots).result().get_counts()
        fids.append(float(np.clip(counts.get(ret_state, 0) / shots, 0, 1)))

    if len(fids) < 2:
        return fids[0] if fids else 0.0
    poly = np.polynomial.Polynomial.fit(np.array(scale_factors, dtype=float), np.array(fids), deg=len(scale_factors) - 1)
    return float(np.clip(poly(0.0), 0, 1))

def zne_dm_fidelity(qc_ansatz, psi_target, dm_sim, basis_gates=None, coupling_map=None,
                    scale_factors=(1, 2, 3)):
    """DM-level ZNE: fold the ansatz, evaluate exact DM fidelity at each scale
    factor, Richardson-extrapolate to zero noise. Fully in-house."""
    fids = [exact_dm_fidelity(fold_circuit(qc_ansatz, sf), psi_target, dm_sim,
                              basis_gates, coupling_map) for sf in scale_factors]
    if len(fids) < 2:
        return fids[0] if fids else 0.0
    poly = np.polynomial.Polynomial.fit(np.array(scale_factors, dtype=float),
                                        np.array(fids), deg=len(scale_factors) - 1)
    return float(np.clip(poly(0.0), 0, 1)), fids

def exact_dm_echo(qc_ansatz, target_inv, dm_sim, basis_gates=None, coupling_map=None, probe="zero"):
    """Inversion-test echo P(|0..0>) evaluated from the exact noisy density
    matrix of U_ansatz · U_target†. Returns (p_00, rho) so callers can also
    extract the purity needed by VD."""
    qc = qc_ansatz.compose(target_inv)
    qc = _prepend_probe(qc, probe)
    if probe == "plus":
        qc.h(range(N_QUBITS))
    if basis_gates is None:
        basis_gates = _get_basis_gates(FakeFez())
    cm = _get_coupling_map(dm_sim) if coupling_map is None else coupling_map
    qc_t = transpile(qc, coupling_map=cm, basis_gates=basis_gates,
                     initial_layout=list(range(N_QUBITS)), optimization_level=3)
    qc_t.save_density_matrix()
    rho = np.array(dm_sim.run(qc_t, shots=1).result().data(0)['density_matrix'])
    p00 = float(np.clip(np.real(rho[0, 0]), 0, 1))
    return p00, rho

def cdr_echo_mitigated(qc_ansatz, opt_params, target_inv, dm_sim,
                       basis_gates=None, coupling_map=None, n_cdr=20,
                       rng_seed=7, probe="zero", print_fn=None):
    """Clifford Data Regression for the inversion-test echo, fully in-house.

    Training circuits share the ansatz's 21-term product structure with each
    effective angle theta_j = rho_n[j] * tau[j] drawn 50/50 from {0, pi/2, pi}
    (exactly Clifford) or kept at the optimized value (near-Clifford). Ideal and
    noisy echoes come from noiseless_echo_fidelity / exact_dm_echo respectively;
    a linear model ideal = a*noisy + b is fit by least squares and applied to the
    actual noisy echo. Returns (corrected_echo, r2)."""
    rng = np.random.default_rng(rng_seed)
    rho_w = np.maximum(opt_params[:N_TERMS], 1e-6)
    rho_n = rho_w / rho_w.sum()
    theta = rho_n * opt_params[N_TERMS:]
    noisy_x, ideal_y = [], []
    for _ in range(n_cdr):
        ang = np.where(rng.random(N_TERMS) < 0.5,
                       rng.integers(0, 3, N_TERMS) * np.pi / 2.0,
                       theta)
        qc_t = QuantumCircuit(N_QUBITS)
        for j, t in enumerate(_TERM_INFO):
            qc_t.unitary(la.expm(-1j * ang[j] * t['P2q']), [t['qk_qa'], t['qk_qb']])
        y_ideal = noiseless_echo_fidelity(qc_t, target_inv, probe=probe)
        x_noisy, _ = exact_dm_echo(qc_t, target_inv, dm_sim, basis_gates, coupling_map, probe=probe)
        noisy_x.append(x_noisy)
        ideal_y.append(y_ideal)
    X = np.array(noisy_x)
    Y = np.array(ideal_y)
    a, b = np.polyfit(X, Y, 1)
    resid = Y - (a * X + b)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    f_noisy, _ = exact_dm_echo(qc_ansatz, target_inv, dm_sim, basis_gates, coupling_map, probe=probe)
    f_cdr = float(np.clip(a * f_noisy + b, 0, 1))
    if print_fn is not None:
        print_fn(f"CDR: n_train={len(X)} r2={r2:.4f} noisy_echo={f_noisy:.4f} corrected_echo={f_cdr:.4f}")
    return f_cdr, r2

def vd_echo_fidelity(qc_ansatz, target_inv, dm_sim, basis_gates=None, coupling_map=None, probe="zero"):
    """Virtual distillation (2-copy) echo via the second-order (distillation)
    estimator: f2 = (rho^2)[0,0] / Tr(rho^2) = Tr(rho^2 P0) / Tr(rho^2) with
    P0 = |0..0><0..0|, capped at the circuit's noiseless echo. This is the exact
    2-copy virtual-distillation overlap for the inversion-test observable. Fully
    in-house, computed from the exact DM (no measurement noise)."""
    f_std, rho = exact_dm_echo(qc_ansatz, target_inv, dm_sim, basis_gates, coupling_map, probe=probe)
    rho2 = rho @ rho
    tr2 = float(np.real(np.trace(rho2)))
    purity = float(np.clip(tr2, 0, 1))
    nl = noiseless_echo_fidelity(qc_ansatz, target_inv, probe=probe)
    if tr2 > 1e-12:
        f_vd = float(np.real(rho2[0, 0])) / tr2
    else:
        f_vd = f_std
    f_vd = float(min(np.clip(f_vd, 0.0, 1.0), nl))
    return f_vd, purity, f_std

def fold_circuit_standard(qc, scale_factor):
    """Standard IBM-style ZNE folding with an explicit odd scale factor.

    Each gate U is mapped to U (U^+ U)^((s-1)/2) -- `scale_factor` copies
    alternating inverse/original -- preserving total unitarity. `scale_factor`
    is the literal odd noise amplification (1, 3, 5, ...).
    """
    copies = int(scale_factor)
    if copies == 1:
        return qc.copy()
    if copies % 2 == 0:
        copies += 1
    folded = QuantumCircuit(*qc.qregs, *qc.cregs)
    for item in qc.data:
        instr = item.operation
        if instr.name in ('measure', 'barrier', 'reset'):
            folded.append(instr, item.qubits, item.clbits)
            continue
        for i in range(copies):
            op = instr.inverse() if i % 2 else instr
            folded.append(op, item.qubits, item.clbits)
    return folded

def ibm_zne_echo_fidelity(qc_ansatz, target_inv, meas_backend, shots, label,
                          scale_factors=(1, 3, 5), physical_layout=None, probe="zero",
                          ceiling=None):
    """Standard IBM-style ZNE: unitary folding at odd scale factors with all
    scale factors batched into ONE job, then exponential extrapolation (IBM
    default) with polynomial fallback. Returns (raw_echo, fids, zne)."""
    from qiskit_ibm_runtime import SamplerV2
    pubs = []
    metas = []
    for sf in scale_factors:
        qc = qc_ansatz.compose(target_inv)
        qc = _prepend_probe(qc, probe)
        if probe == "plus":
            qc.h(range(N_QUBITS))
        qc_f = fold_circuit_standard(qc, sf)
        qc_f.measure_all()
        if physical_layout is not None:
            qc_t = transpile(qc_f, backend=meas_backend, initial_layout=physical_layout, optimization_level=3)
        else:
            basis_gates = _get_basis_gates(FakeFez())
            cm = _get_coupling_map(meas_backend)
            qc_t = transpile(qc_f, coupling_map=cm, basis_gates=basis_gates,
                             initial_layout=list(range(N_QUBITS)), optimization_level=3)
        pubs.append((qc_t,))
        metas.append(sf)
    print(f"{ts()} [{label}] Standard IBM-style ZNE: {len(pubs)} pubs (SF={scale_factors}), shots={shots}")
    job = SamplerV2(mode=meas_backend).run(pubs, shots=shots)
    counts_list = [j.data.meas.get_counts() for j in job.result()]
    ret_state = '0' * N_QUBITS
    fids = [float(np.clip(c.get(ret_state, 0) / shots, 0, 1)) for c in counts_list]
    for sf, fid in zip(metas, fids):
        print(f"{ts()} [{label}] SF={sf}: echo={fid:.6f}")

    xs = np.asarray(list(metas), dtype=float)
    ys = np.asarray(fids, dtype=float)
    if len(xs) < 2:
        return float(fids[0]), fids, float(fids[0])
    monotone = all(ys[i] >= ys[i + 1] for i in range(len(ys) - 1))
    if not monotone:
        print(f"{ts()} [{label}] WARNING: fidelities non-monotonic vs scale factor "
              f"({fids}) -- near the noise floor; ZNE extrapolation unreliable.")
    exp0 = float(np.mean(ys))
    try:
        ys_c = np.clip(ys, 1e-12, None)
        cfit = np.polynomial.polynomial.polyfit(xs, np.log(ys_c), 1)
        exp0 = float(np.exp(cfit[0]))
    except Exception:
        pass
    if len(xs) >= 3:
        c = np.polynomial.polynomial.polyfit(xs[-3:], ys[-3:], 2)
        poly0 = float(c[0])
    else:
        c = np.polynomial.polynomial.polyfit(xs, ys, len(xs) - 1)
        poly0 = float(c[0])
    zne = exp0 if (0.0 <= exp0 <= 1.0) else poly0
    if ceiling is not None:
        zne = min(zne, ceiling)
    print(f"{ts()} [{label}] ZNE fit: exp(0)={exp0:.6f}  poly/Richardson(0)={poly0:.6f}  -> zne_echo={zne:.6f}")
    return float(fids[0]), fids, float(zne)

# ==============================================================================
# PIPELINES & HARDWARE INTERFACE
# ==============================================================================
def get_ibm_service(token="5IYrwUNE5zyh64upXx-R_f9IrpGOnvjVsbhKpfnP81-L"):
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = token or os.environ.get('IBMQ_TOKEN') or os.environ.get('IBM_QUANTUM_TOKEN')
    if token and token != "YOUR_API_KEY_HERE":
        for channel in ['ibm_quantum_platform', 'ibm_cloud', 'ibm_quantum']:
            try:
                return QiskitRuntimeService(channel=channel, token=token)
            except Exception:
                pass
        raise ValueError(
            f"Unable to connect to IBM Quantum with the provided token (token ends in ...{token[-6:]}).\n"
            "This usually means the API token has expired or is invalid.\n"
            "Please generate a new API token at https://quantum.ibm.com and pass it via --token <KEY> or export IBMQ_TOKEN=<KEY>."
        )
    try:
        return QiskitRuntimeService()
    except Exception as e:
        raise ValueError(
            "No valid IBM Quantum API token found.\n"
            "Please provide your active API token from https://quantum.ibm.com using:\n"
            "  --token <YOUR_TOKEN>\n"
            "or by setting the environment variable:\n"
            "  export IBMQ_TOKEN=\"<YOUR_TOKEN>\""
        ) from e

@dataclass
class SessionResult:
    date: str
    backend_name: str
    backend_version: str
    quietest_path: list
    target_seed: int
    mode: str
    probe: str
    n_optim_iters: int
    opt_dm_fidelity: float
    target_2q: int
    target_depth: int
    ansatz_2q: int
    ansatz_depth: int
    noiseless_echo: float
    raw_echo: float
    raw_echo_error: float
    tpnm_echo: float
    zne_echo: float
    zne_dm: Optional[float] = None
    cdr_echo: Optional[float] = None
    cdr_r2: Optional[float] = None
    vd_echo: Optional[float] = None
    vd_purity: Optional[float] = None
    opt_params: Optional[list] = None

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump({k: v for k, v in asdict(self).items() if v is not None}, f, indent=2, default=str)
        print(f"{ts()} Saved results to {filename}")

def run_simulation_benchmark(n_seeds=1, target_seed=0, modes=("per_pair",),
                             max_iters=30, shots=4000, probe="zero", output_dir="results", do_zne=True,
                             optimise=True, do_cdr=False, do_vd=False, n_cdr=20, cdr_seed=7):
    sim_dm, sim_meas, cmap = get_default_sims()
    fb = FakeFez()
    bg = _get_basis_gates(fb)
    quietest_path = get_quietest_fake_path()

    print(f"{ts()} Building TPNM calibration (sim)...")
    tpnm_mat = build_tpnm_calibration(sim_meas, cmap, bg, shots=2000)

    psi_target, coeffs, target_qc = build_target(seed=target_seed)
    target_inv = build_target_inv_circuit(coeffs)
    target_2q, target_depth = count_2q_gates(target_qc, cmap, bg)

    all_results = []
    for seed in range(n_seeds):
        for mode in modes:
            print(f"\n{'=' * 60}\n{ts()} Seed {seed + 1}/{n_seeds} | mode={mode}\n{'=' * 60}")
            p0 = init_params(seed=seed)
            if optimise or mode != "product":
                opt_params = run_dual_mma(p0, psi_target, sim_dm, use_upte=_use_upte_flag(mode), max_iters=max_iters)
            else:
                opt_params = baseline_params(seed=target_seed)
                print(f"{ts()} Untuned product-Trotter baseline (random params, seed {target_seed + 42}); no optimisation.")
            ans_qc = build_ansatz(opt_params, mode=mode)
            dm_fid = exact_dm_fidelity(ans_qc, psi_target, sim_dm, bg, cmap)

            nl = noiseless_echo_fidelity(ans_qc, target_inv, probe=probe)
            raw = dfe_fidelity(ans_qc, target_inv, sim_meas, shots, label=mode, probe=probe)
            raw_err = np.sqrt(raw * (1.0 - raw) / shots)
            tpn = inversion_test_mitigated(ans_qc, target_inv, sim_meas, cmap, bg, tpnm_mat, shots, label=f"{mode}_mit", probe=probe)
            zne = zne_echo_fidelity(ans_qc, target_inv, sim_meas, shots, label=f"{mode}_zne", probe=probe) if do_zne else 0.0

            zne_dm = None
            if do_zne:
                zne_dm, _ = zne_dm_fidelity(ans_qc, psi_target, sim_dm, bg, cmap)
            cdr_echo, cdr_r2 = None, None
            if do_cdr:
                cdr_echo, cdr_r2 = cdr_echo_mitigated(
                    ans_qc, opt_params, target_inv, sim_dm, bg, cmap,
                    n_cdr=n_cdr, rng_seed=cdr_seed + seed, probe=probe,
                    print_fn=lambda m: print(f"{ts()} {m}"))
            vd_echo, vd_purity = None, None
            if do_vd:
                vd_echo, vd_purity, _ = vd_echo_fidelity(ans_qc, target_inv, sim_dm, bg, cmap, probe=probe)

            ans_2q, ans_depth = count_2q_gates(ans_qc, cmap, bg)
            res = SessionResult(
                date=datetime.now().isoformat(), backend_name="FakeFez", backend_version="sim",
                quietest_path=list(quietest_path), target_seed=target_seed, mode=mode, probe=probe,
                n_optim_iters=max_iters, opt_dm_fidelity=float(dm_fid), target_2q=int(target_2q),
                target_depth=int(target_depth), ansatz_2q=int(ans_2q), ansatz_depth=int(ans_depth),
                noiseless_echo=float(nl), raw_echo=float(raw), raw_echo_error=float(raw_err),
                tpnm_echo=float(tpn), zne_echo=float(zne), zne_dm=zne_dm,
                cdr_echo=cdr_echo, cdr_r2=cdr_r2, vd_echo=vd_echo, vd_purity=vd_purity,
                opt_params=opt_params.tolist()
            )
            res.save(os.path.join(output_dir, f"sim_{mode}_seed{seed}_{datetime.now():%Y%m%d_%H%M%S}.json"))
            all_results.append(res)
    return all_results

def run_lpte_experiment(backend_name="ibm_fez", target_seed=0, modes=("per_pair",),
                        optim_iters=40, inv_shots=4000, tpnm_cal_shots=5000, output_dir="results",
                        probe="zero", do_zne=True, optimise=True, token=None):
    service = get_ibm_service(token=token)
    real_backend = service.backend(backend_name)
    real_cmap = _get_coupling_map(real_backend)
    real_path = find_quietest_path(real_backend)
    bg = _get_basis_gates(real_backend)

    psi_target, coeffs, target_qc = build_target(seed=target_seed)
    target_inv = build_target_inv_circuit(coeffs)
    target_2q, target_depth = count_2q_gates(target_qc, real_cmap, bg)

    tpnm_matrix = build_tpnm_calibration(real_backend, real_cmap, bg, shots=tpnm_cal_shots, physical_layout=real_path)
    sim_dm, _, _ = get_default_sims()

    for mode in modes:
        print(f"\n{'=' * 60}\n{ts()} Mode: {mode}\n{'=' * 60}")
        p0 = init_params(seed=42)
        if optimise or mode != "product":
            opt_params = run_dual_mma(p0, psi_target, sim_dm, use_upte=_use_upte_flag(mode), max_iters=optim_iters)
        else:
            opt_params = baseline_params(seed=target_seed)
            print(f"{ts()} Untuned product-Trotter baseline (random params, seed {target_seed + 42}); no optimisation.")
        ans_qc = build_ansatz(opt_params, mode=mode)
        dm_fid = exact_dm_fidelity(ans_qc, psi_target, sim_dm, bg, None)

        nl = noiseless_echo_fidelity(ans_qc, target_inv, probe=probe)
        raw = dfe_fidelity(ans_qc, target_inv, real_backend, inv_shots, label=mode, physical_layout=real_path, probe=probe)
        raw_err = np.sqrt(raw * (1.0 - raw) / inv_shots)
        tpn = inversion_test_mitigated(ans_qc, target_inv, real_backend, real_cmap, bg, tpnm_matrix, inv_shots, label=f"{mode}_mit", physical_layout=real_path, probe=probe)
        zne = zne_echo_fidelity(ans_qc, target_inv, real_backend, inv_shots, label=f"{mode}_zne", physical_layout=real_path, probe=probe) if do_zne else 0.0

        ans_2q, ans_depth = count_2q_gates(ans_qc, real_cmap, bg)
        res = SessionResult(
            date=datetime.now().isoformat(), backend_name=backend_name, backend_version="hardware",
            quietest_path=list(real_path), target_seed=target_seed, mode=mode, probe=probe,
            n_optim_iters=optim_iters, opt_dm_fidelity=float(dm_fid), target_2q=int(target_2q),
            target_depth=int(target_depth), ansatz_2q=int(ans_2q), ansatz_depth=int(ans_depth),
            noiseless_echo=float(nl), raw_echo=float(raw), raw_echo_error=float(raw_err),
            tpnm_echo=float(tpn), zne_echo=float(zne), opt_params=opt_params.tolist()
        )
        res.save(os.path.join(output_dir, f"hw_{backend_name}_{mode}_{datetime.now():%Y%m%d_%H%M%S}.json"))

def run_zne_only_experiment(backend_name="ibm_fez", target_seed=0, shots=4000,
                            noise_factors=(1, 3, 5), probe="zero", token=None,
                            params_json=None, output_dir="results"):
    """ZNE-only pass: NO TOPAZ re-optimization. Loads saved opt_params, rebuilds
    the LPTE ansatz and target-inverse, runs standard IBM-style ZNE on hardware."""
    if not params_json or not os.path.exists(params_json):
        raise ValueError(f"--zne-only requires --params <path to saved 8q results JSON> (got: {params_json})")
    with open(params_json) as f:
        saved = json.load(f)
    opt_params = np.asarray(saved["opt_params"], dtype=float)
    if opt_params.shape != (N_PARAMS,):
        raise ValueError(f"Expected {N_PARAMS} params in {params_json}, got {opt_params.shape}")

    service = get_ibm_service(token=token)
    real_backend = service.backend(backend_name)
    real_cmap = _get_coupling_map(real_backend)
    real_path = find_quietest_path(real_backend)
    bg = _get_basis_gates(real_backend)

    psi_target, coeffs, target_qc = build_target(seed=target_seed)
    target_inv = build_target_inv_circuit(coeffs)
    target_2q, target_depth = count_2q_gates(target_qc, real_cmap, bg)

    ans_qc = build_ansatz(opt_params, mode=saved.get("mode", "per_pair"))
    ans_2q, ans_depth = count_2q_gates(ans_qc, real_cmap, bg)

    nl = noiseless_echo_fidelity(ans_qc, target_inv, probe=probe)
    print(f"{ts()} Noiseless echo (reference): {nl:.6f}")

    raw, fids, zne = ibm_zne_echo_fidelity(
        ans_qc, target_inv, real_backend, shots,
        label=f"{saved.get('mode', 'per_pair')}_standard_zne", scale_factors=noise_factors,
        physical_layout=real_path, probe=probe)
    raw_err = float(np.sqrt(raw * (1.0 - raw) / shots))

    res = SessionResult(
        date=datetime.now().isoformat(), backend_name=backend_name,
        backend_version="hardware_8q_zne_only",
        quietest_path=list(real_path), target_seed=target_seed,
        mode=saved.get("mode", "per_pair"), probe=probe,
        n_optim_iters=0, opt_dm_fidelity=float(saved.get("opt_dm_fidelity", nl)),
        target_2q=int(target_2q), target_depth=int(target_depth),
        ansatz_2q=int(ans_2q), ansatz_depth=int(ans_depth),
        noiseless_echo=float(nl), raw_echo=float(raw), raw_echo_error=raw_err,
        tpnm_echo=0.0, zne_echo=float(zne), opt_params=opt_params.tolist()
    )
    res.save(os.path.join(output_dir, f"hw_{backend_name}_zne_only_{datetime.now():%Y%m%d_%H%M%S}.json"))
    print(f"{ts()} ZNE-only complete: raw_echo={raw:.6f}  zne_echo={zne:.6f}  noiseless_echo={nl:.6f}")

# ==============================================================================
# MAIN CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fez 8-qubit state-preservation test.")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--mode", default="per_pair", choices=["all", "product", "per_pair", "global"])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--probe", default="zero", choices=["zero", "plus"])
    parser.add_argument("--no-zne", action="store_true")
    parser.add_argument("--no-opt", action="store_true",
                        help="product mode: measure untuned (random, unoptimised) baseline instead of MMA-optimising")
    parser.add_argument("--output", default="results")
    parser.add_argument("--lpte", action="store_true")
    parser.add_argument("--backend", default="ibm_fez")
    parser.add_argument("--tpnm-shots", type=int, default=5000)
    parser.add_argument("--target-seed", type=int, default=0)
    parser.add_argument("--token", default=None, help="IBM Quantum API token")
    parser.add_argument("--zne-only", action="store_true",
                        help="Run standard IBM-style ZNE on a saved circuit (no TOPAZ re-optimization); requires --params.")
    parser.add_argument("--params", default=None,
                        help="Path to a saved 8q results JSON containing opt_params (required with --zne-only).")
    parser.add_argument("--noise-factors", default="1,3,5",
                        help="Comma-separated odd ZNE scale factors (default 1,3,5).")
    parser.add_argument("--cdr", action="store_true",
                        help="Add self-contained Clifford Data Regression echo correction (simulation).")
    parser.add_argument("--vd", action="store_true",
                        help="Add self-contained Virtual Distillation (2-copy) echo correction (simulation).")
    parser.add_argument("--n-cdr", type=int, default=20,
                        help="Number of Clifford training circuits for CDR (default 20).")
    args = parser.parse_args()

    modes = ("product", "per_pair", "global") if args.mode == "all" else (args.mode,)
    if args.zne_only:
        nf = tuple(int(x) for x in args.noise_factors.split(","))
        run_zne_only_experiment(backend_name=args.backend, target_seed=args.target_seed, shots=args.shots,
                                noise_factors=nf, probe=args.probe, token=args.token,
                                params_json=args.params, output_dir=args.output)
        return
    if args.lpte:
        run_lpte_experiment(backend_name=args.backend, target_seed=args.target_seed, modes=modes,
                            optim_iters=args.iters, inv_shots=args.shots, tpnm_cal_shots=args.tpnm_shots,
                            output_dir=args.output, probe=args.probe, do_zne=not args.no_zne,
                            optimise=not args.no_opt, token=args.token)
    else:
        run_simulation_benchmark(n_seeds=args.seeds, target_seed=args.target_seed, modes=modes,
                                 max_iters=args.iters, shots=args.shots, probe=args.probe,
                                 output_dir=args.output, do_zne=not args.no_zne,
                                 optimise=not args.no_opt,
                                 do_cdr=args.cdr, do_vd=args.vd, n_cdr=args.n_cdr)

if __name__ == "__main__":
    main()