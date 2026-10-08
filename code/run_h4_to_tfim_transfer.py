#!/usr/bin/env python3
"""Cross-hamiltonian TOPAZ transfer test: H4-learned layout -> 8-qubit TFIM.

Fully self-contained: the entire H4 LPTE stack (Hamiltonian, noise model, LPTE
circuit builder, dual MMA optimizer, 3-stage TOPAZ pipeline, QEM utilities) is
inlined here so this file has zero custom-module imports.

Phase 1: Run the existing H4 LPTE VQE pipeline, capture the optimized TOPAZ
         parameter vector and gate count as LAYOUT A.
Phase 2A (transfer): Build an 8-qubit TFIM. Freeze TOPAZ at layout A, optimize
         ONLY the angle (non-TOPAZ) parameters via MMA, measure + record.
Phase 2B (baseline): Reset the TFIM fresh, run the FULL pipeline (TOPAZ + angles)
         from scratch -> LAYOUT B. Measure + record.
Compare layout A (transfer) vs layout B (from-scratch): energy, delta-FCI, gates.
"""

import argparse
import json
import os
import sys
import time
import warnings
import tempfile
import numpy as np
import scipy.linalg as la
import networkx as nx

warnings.filterwarnings("ignore")

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez

# ── Per-script log file (clears any previous log) ──
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{os.path.splitext(os.path.basename(__file__))[0]}_log.txt")
if os.path.exists(log_path):
    os.remove(log_path)

def ts() -> str:
    return time.strftime("[%H:%M:%S]")

def LOG(msg: str):
    line = f"{ts()} {msg}"
    print(line)
    with open(log_path, "a") as f:
        f.write(line + "\n")

# ── Base Constants & Pauli Helpers ──
H4_CHAIN = "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22"
N_QUBITS = 8
N_PAIRS = N_QUBITS - 1  # 7 pairs

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

_PAIRS_2Q = {
    'XX': np.kron(_X, _X), 'XY': np.kron(_X, _Y), 'XZ': np.kron(_X, _Z),
    'YX': np.kron(_Y, _X), 'YY': np.kron(_Y, _Y), 'YZ': np.kron(_Y, _Z),
    'ZX': np.kron(_Z, _X), 'ZY': np.kron(_Z, _Y), 'ZZ': np.kron(_Z, _Z),
}
NN_LAYER_1 = ['XX', 'YY', 'ZZ']
NN_LAYER_2 = ['XY', 'YX', 'ZX']

# ── 1. H4 Hamiltonian Builder ──
_HAS_NATURE = False
try:
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.circuit.library import HartreeFock
    from qiskit_nature.units import DistanceUnit
    _HAS_NATURE = True
except ImportError:
    pass

def build_h4_hamiltonian() -> dict:
    if not _HAS_NATURE:
        raise ImportError("qiskit-nature is required to build H₄ Hamiltonian.")
    driver = PySCFDriver(atom=H4_CHAIN, basis="sto3g", unit=DistanceUnit.ANGSTROM)
    problem = driver.run()
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())
    H_dense = qubit_op.to_matrix(sparse=True).toarray().astype(complex)
    E_nuc = problem.nuclear_repulsion_energy
    H_total = H_dense + E_nuc * np.eye(2**N_QUBITS, dtype=complex)
    fci_energy = float(np.min(la.eigvalsh(H_total)))
    hf_circuit = HartreeFock(num_spatial_orbitals=problem.num_spatial_orbitals,
                             num_particles=problem.num_particles,
                             qubit_mapper=mapper)
    return {
        "HAMILTONIAN_DENSE": H_dense,
        "E_NUC": E_nuc,
        "FCI_ENERGY": fci_energy,
        "HF_CIRCUIT": hf_circuit,
    }

# ── 2. Noise Setup ──
def find_quietest_8q_path() -> list[int]:
    backend = FakeFez()
    props = backend.properties()
    bg = backend.configuration().basis_gates
    tq = [g for g in bg if g in ('cx', 'cz', 'ecr')][0]
    coupling_map = backend.configuration().coupling_map

    def _gerr(q1, q2):
        try: return props.gate_error(tq, [q1, q2]) or 0.0
        except Exception: return 0.0
    def _rerr(q):
        try: return props.readout_error(q) or 0.0
        except Exception: return 0.0

    G = nx.Graph()
    for q1, q2 in coupling_map:
        weight = (_gerr(q1, q2) + _gerr(q2, q1)) / 2 + 0.1 * (_rerr(q1) + _rerr(q2)) / 2
        G.add_edge(q1, q2, weight=weight)

    paths = []
    def dfs(n, p):
        if len(p) == N_QUBITS:
            paths.append(p)
            return
        for nb in G.neighbors(n):
            if nb not in p:
                dfs(nb, p + [nb])
    for n in G.nodes:
        dfs(n, [n])

    scored = [(sum(G[p[i]][p[i+1]]['weight'] for i in range(N_QUBITS - 1)), p) for p in paths]
    scored.sort(key=lambda x: x[0])
    return scored[0][1]

def build_noise_model() -> tuple:
    quietest_path = find_quietest_8q_path()
    backend = FakeFez()
    fc = backend.configuration().coupling_map
    bg = backend.configuration().basis_gates
    p2v = {p: i for i, p in enumerate(quietest_path)}
    red_edges = [(p2v[a], p2v[b]) for a, b in fc if a in quietest_path and b in quietest_path]
    cmap = CouplingMap(red_edges)

    from qiskit_aer.noise import NoiseModel
    fn = NoiseModel.from_backend(backend)
    nm = NoiseModel(basis_gates=bg)

    for phys_q in quietest_path:
        vq = p2v[phys_q]
        ro = fn._local_readout_errors.get(phys_q)
        if ro:
            nm.add_readout_error(ro, [vq])

    for gn in bg:
        for qubits, error in fn._local_quantum_errors.get(gn, {}).items():
            nq = tuple(p2v[q] for q in qubits if q in quietest_path)
            if len(nq) == len(qubits):
                nm.add_quantum_error(error, gn, nq)

    return nm, cmap, bg, quietest_path

# ── 3. LPTE Circuit Builders ──
def _pauli_expm(theta: float, P: np.ndarray) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return c * np.eye(4, dtype=complex) - 1j * s * P

def _smooth_rho_tau(rho_raw: np.ndarray, tau_raw: np.ndarray):
    rho_w = np.exp(rho_raw).reshape(N_PAIRS, 3)
    rho_n = rho_w / rho_w.sum(axis=1, keepdims=True)
    rho_n = rho_n.flatten()
    tau = np.pi * (0.5 + 0.5 * np.tanh(tau_raw))
    return rho_n, tau

def build_lpte_layer(topaz_params: np.ndarray, layer_num: int, active_pairs: list[bool] = None) -> QuantumCircuit:
    nc = 3 * N_PAIRS
    rho_n, tau = _smooth_rho_tau(topaz_params[:nc], topaz_params[nc:])
    pauli_set = NN_LAYER_1 if layer_num == 1 else NN_LAYER_2
    qc = QuantumCircuit(N_QUBITS)
    for pi in range(N_PAIRS):
        if active_pairs is not None and not active_pairs[pi]:
            continue
        qa, qb = pi, pi + 1
        B = np.zeros((4, 4), dtype=complex)
        for pp, pauli_name in enumerate(pauli_set):
            j = pi * 3 + pp
            B += rho_n[j] * _pauli_expm(tau[j], _PAIRS_2Q[pauli_name])
        Us, _, Vhs = la.svd(B)
        qc.unitary(Us @ Vhs, [qa, qb])
    return qc

N_ANGLES_PER_LAYER = N_QUBITS * 2  # Ry + Rz
N_TOPAZ_LAYER = 2 * 3 * N_PAIRS
N_TOPAZ = 2 * N_TOPAZ_LAYER
N_ANGLES = 2 * N_ANGLES_PER_LAYER

def build_dual_lpte_circuit(topaz_params: np.ndarray, angle_params: np.ndarray = None,
                            hf_circuit: QuantumCircuit = None, active_pairs: list[bool] = None) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)
    if hf_circuit is not None:
        qc.compose(hf_circuit, inplace=True)
    for layer_idx in range(2):
        if angle_params is not None:
            off = layer_idx * N_ANGLES_PER_LAYER
            for q in range(N_QUBITS):
                qc.ry(angle_params[off + q], q)
                qc.rz(angle_params[off + N_QUBITS + q], q)
        layer_params = topaz_params[layer_idx * N_TOPAZ_LAYER : (layer_idx + 1) * N_TOPAZ_LAYER]
        qc.compose(build_lpte_layer(layer_params, layer_idx + 1, active_pairs), inplace=True)
    return qc

# ── 4. MMA Optimizer ──
class MMAOptimizer:
    def __init__(self, n: int, move_limit: float = 0.4, gamma: float = 0.5):
        self.n = n
        self.move_limit = move_limit
        self.gamma = gamma
        self.L = None
        self.U = None
        self.prev_loss = None

    def init(self, p0: np.ndarray, delta: float = 0.6):
        self.L = p0 - delta
        self.U = p0 + delta

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        x_new = np.zeros_like(x)
        for i in range(self.n):
            g = grad[i]
            xi = x[i]
            Li = self.L[i]
            Ui = self.U[i]
            pi = abs(g) * (Ui - xi) ** 2 if g < 0 else 0.0
            qi = abs(g) * (xi - Li) ** 2 if g >= 0 else 0.0
            denom = pi / (Ui - xi + 1e-12) ** 2 + qi / (xi - Li + 1e-12) ** 2
            if denom > 1e-12:
                delta_i = pi / (Ui - xi + 1e-12) - qi / (xi - Li + 1e-12)
                x_new[i] = xi + delta_i / denom
            else:
                x_new[i] = xi
            x_new[i] = np.clip(x_new[i], max(xi - self.move_limit, Li + 1e-6), min(xi + self.move_limit, Ui - 1e-6))
        return x_new

    def update(self, x: np.ndarray, loss: float):
        good = self.prev_loss is None or loss < self.prev_loss - 1e-8
        s = 1.2 / self.gamma if good else self.gamma
        self.L = x - s * (x - self.L)
        self.U = x + s * (self.U - x)
        self.L = np.minimum(self.L, self.U - 1e-4)
        self.U = np.maximum(self.U, self.L + 1e-4)
        self.prev_loss = loss

# ── 5. LPTE Loss & VQE Optimization ──
def reg_penalty_lpte(topaz_params: np.ndarray) -> float:
    total_penalty = 0.0
    for layer_idx in range(2):
        layer_params = topaz_params[layer_idx * N_TOPAZ_LAYER : (layer_idx + 1) * N_TOPAZ_LAYER]
        rho_n, tau = _smooth_rho_tau(layer_params[:N_TOPAZ_LAYER//2], layer_params[N_TOPAZ_LAYER//2:])
        pauli_set = NN_LAYER_1 if layer_idx == 0 else NN_LAYER_2
        for pi in range(N_PAIRS):
            B = np.zeros((4, 4), dtype=complex)
            for pp, pauli_name in enumerate(pauli_set):
                j = pi * 3 + pp
                B += rho_n[j] * _pauli_expm(tau[j], _PAIRS_2Q[pauli_name])
            B_dag_B = B.conj().T @ B
            dev = B_dag_B - np.eye(4, dtype=complex)
            total_penalty += float(np.real(np.trace(dev.conj().T @ dev)))
    return total_penalty

def noisy_energy(build_circuit_fn, params, H_den, sim_dm, cm, basis_gates) -> float:
    qc = build_circuit_fn(*params) if isinstance(params, tuple) else build_circuit_fn(params)
    qc_t = transpile(qc, coupling_map=cm, basis_gates=basis_gates, optimization_level=3)
    qc_t.save_density_matrix()
    rho = np.array(sim_dm.run(qc_t, shots=1).result().data(0)["density_matrix"])
    return float(np.real(np.trace(H_den @ rho)))

def noiseless_energy(build_circuit_fn, params, H_den, sv_sim, cm, basis_gates) -> float:
    qc = build_circuit_fn(*params) if isinstance(params, tuple) else build_circuit_fn(params)
    qc_t = transpile(qc, basis_gates=basis_gates, optimization_level=0)
    qc_t.save_statevector()
    sv = np.array(sv_sim.run(qc_t, shots=1).result().data(0)["statevector"])
    return float(np.real(sv.conj() @ H_den @ sv))

def energy(topaz_params: np.ndarray, angle_params: np.ndarray, build_fn, H_den, sim_dm, cm, bg, active_pairs=None, reg=0.001) -> float:
    E_noisy = noisy_energy(build_fn, (topaz_params, angle_params, active_pairs), H_den, sim_dm, cm, bg)
    return E_noisy + reg * reg_penalty_lpte(topaz_params)

def topaz_gradient(topaz_params, angle_params, build_fn, H_den, sim_dm, cm, bg, active_pairs, active_mask, prev_grad, reg=0.001):
    grad = np.zeros(N_TOPAZ)
    alpha = 0.4
    for i in range(N_TOPAZ):
        if not active_mask[i]: continue
        eps_i = 0.01 if (i % N_TOPAZ_LAYER) < (N_TOPAZ_LAYER // 2) else np.pi / 16
        p = topaz_params.copy(); p[i] += eps_i
        loss_plus = energy(p, angle_params, build_fn, H_den, sim_dm, cm, bg, active_pairs=active_pairs, reg=reg)
        p = topaz_params.copy(); p[i] -= eps_i
        loss_minus = energy(p, angle_params, build_fn, H_den, sim_dm, cm, bg, active_pairs=active_pairs, reg=reg)
        grad[i] = (loss_plus - loss_minus) / (2 * eps_i)
    grad = alpha * grad + (1 - alpha) * prev_grad
    grad[~active_mask] = 0
    return grad

def angle_gradient(topaz_params, angle_params, build_fn, H_den, sim_dm, cm, bg, active_pairs, active_mask, prev_grad, reg=0.001):
    grad = np.zeros(N_ANGLES)
    alpha = 0.4
    eps = np.pi / 16
    for i in range(N_ANGLES):
        a = angle_params.copy(); a[i] += eps
        loss_plus = energy(topaz_params, a, build_fn, H_den, sim_dm, cm, bg, active_pairs=active_pairs, reg=reg)
        a = angle_params.copy(); a[i] -= eps
        loss_minus = energy(topaz_params, a, build_fn, H_den, sim_dm, cm, bg, active_pairs=active_pairs, reg=reg)
        grad[i] = (loss_plus - loss_minus) / (2 * eps)
    grad = alpha * grad + (1 - alpha) * prev_grad
    return grad

def run_mma_stage(build_fn, init_topaz, init_angles, H_den, sim_dm, cm, bg, n_outer, seed, active_pairs=None, e_nuc: float = 0.0) -> dict:
    topaz = init_topaz.copy()
    angles = init_angles.copy()

    active_mask = np.ones(N_TOPAZ, dtype=bool)
    if active_pairs is not None:
        for p_idx, active in enumerate(active_pairs):
            if not active:
                for layer in range(2):
                    rho_off = layer * N_TOPAZ_LAYER + p_idx * 3
                    tau_off = layer * N_TOPAZ_LAYER + (N_TOPAZ_LAYER // 2) + p_idx * 3
                    topaz[rho_off : rho_off+3] = -10
                    topaz[tau_off : tau_off+3] = -10
                    active_mask[rho_off : tau_off+3] = False

    nc = 3 * N_PAIRS
    mma_r1 = MMAOptimizer(nc, move_limit=0.2); mma_r1.init(topaz[:nc], delta=0.4)
    mma_t1 = MMAOptimizer(nc, move_limit=0.6); mma_t1.init(topaz[nc:N_TOPAZ_LAYER], delta=0.8)
    mma_r2 = MMAOptimizer(nc, move_limit=0.2); mma_r2.init(topaz[N_TOPAZ_LAYER:N_TOPAZ_LAYER+nc], delta=0.4)
    mma_t2 = MMAOptimizer(nc, move_limit=0.6); mma_t2.init(topaz[N_TOPAZ_LAYER+nc:], delta=0.8)

    mma_ang = MMAOptimizer(N_ANGLES, move_limit=0.3); mma_ang.init(angles, delta=0.6)

    cur_loss = energy(topaz, angles, build_fn, H_den, sim_dm, cm, bg, active_pairs)
    best_topaz = topaz.copy()
    best_angles = angles.copy()
    best_loss = cur_loss
    stagnation = 0

    prev_g_t = np.zeros(N_TOPAZ)
    prev_g_a = np.zeros(N_ANGLES)

    for outer_it in range(1, n_outer + 1):
        t_i = time.time()

        grad = topaz_gradient(topaz, angles, build_fn, H_den, sim_dm, cm, bg, active_pairs, active_mask, prev_g_t)
        prev_g_t = grad.copy()

        new_r1 = mma_r1.step(topaz[:nc], grad[:nc])
        new_t1 = mma_t1.step(topaz[nc:N_TOPAZ_LAYER], grad[nc:N_TOPAZ_LAYER])
        new_r2 = mma_r2.step(topaz[N_TOPAZ_LAYER:N_TOPAZ_LAYER+nc], grad[N_TOPAZ_LAYER:N_TOPAZ_LAYER+nc])
        new_t2 = mma_t2.step(topaz[N_TOPAZ_LAYER+nc:], grad[N_TOPAZ_LAYER+nc:])
        new_topaz = np.concatenate([new_r1, new_t1, new_r2, new_t2])
        new_topaz[~active_mask] = topaz[~active_mask]

        new_loss = energy(new_topaz, angles, build_fn, H_den, sim_dm, cm, bg, active_pairs)
        if new_loss < cur_loss - 1e-8:
            topaz = new_topaz
            cur_loss = new_loss
            stagnation = 0
            if cur_loss < best_loss:
                best_topaz, best_angles, best_loss = topaz.copy(), angles.copy(), cur_loss
        else:
            stagnation += 1
            for m in (mma_r1, mma_t1, mma_r2, mma_t2):
                m.move_limit = max(0.02, m.move_limit * 0.8)

        mma_r1.update(topaz[:nc], cur_loss)
        mma_t1.update(topaz[nc:N_TOPAZ_LAYER], cur_loss)
        mma_r2.update(topaz[N_TOPAZ_LAYER:N_TOPAZ_LAYER+nc], cur_loss)
        mma_t2.update(topaz[N_TOPAZ_LAYER+nc:], cur_loss)

        for inner_it in range(5):
            ag = angle_gradient(topaz, angles, build_fn, H_den, sim_dm, cm, bg, active_pairs, active_mask, prev_g_a)
            prev_g_a = ag.copy()

            new_angles = mma_ang.step(angles, ag)
            new_inner_loss = energy(topaz, new_angles, build_fn, H_den, sim_dm, cm, bg, active_pairs)

            if new_inner_loss < cur_loss - 1e-8:
                angles = new_angles
                cur_loss = new_inner_loss
                if cur_loss < best_loss:
                    best_topaz, best_angles, best_loss = topaz.copy(), angles.copy(), cur_loss
            else:
                mma_ang.move_limit = max(0.02, mma_ang.move_limit * 0.8)

            mma_ang.update(angles, cur_loss)

        LOG(f"    Stage iter {outer_it}/{n_outer}: E_total={cur_loss + e_nuc:.6f} Ha [{time.time()-t_i:.1f}s]")

    return {
        "best_topaz": best_topaz,
        "best_angles": best_angles,
        "best_energy": best_loss,
        "success": stagnation <= 20,
    }

def run_full_topaz_pipeline(build_fn, init_topaz, init_angles, H_den, sim_dm, cm, bg, n_outer, seed, e_nuc: float = 0.0):
    LOG("  Stage 1: Continuous Relaxation...")
    res1 = run_mma_stage(build_fn, init_topaz, init_angles, H_den, sim_dm, cm, bg, n_outer, seed, active_pairs=None, e_nuc=e_nuc)
    opt_topaz = res1["best_topaz"]

    LOG("  Stage 2: Hard Pruning...")
    kappas = []
    nc = 3 * N_PAIRS
    for p in range(N_PAIRS):
        tau_layer1 = _smooth_rho_tau(opt_topaz[:nc], opt_topaz[nc:N_TOPAZ_LAYER])[1][p*3:(p+1)*3]
        tau_layer2 = _smooth_rho_tau(opt_topaz[N_TOPAZ_LAYER:N_TOPAZ_LAYER+nc], opt_topaz[N_TOPAZ_LAYER+nc:])[1][p*3:(p+1)*3]
        kappas.append(np.mean([np.sum(np.abs(tau_layer1)), np.sum(np.abs(tau_layer2))]))

    active_pairs = [k >= 0.05 for k in kappas]
    LOG(f"    Pruned {N_PAIRS - sum(active_pairs)} pairs. Survivors: {sum(active_pairs)}")

    LOG("  Stage 3: Sparse Re-optimization...")
    res3 = run_mma_stage(build_fn, res1["best_topaz"], res1["best_angles"], H_den, sim_dm, cm, bg, max(10, n_outer // 2), seed + 100, active_pairs=active_pairs, e_nuc=e_nuc)
    res3["active_pairs"] = active_pairs
    return res3

# ── 6. QEM Utils ──
_BG_LIGHT = None

def fold_circuit(qc_t, sf: int):
    if sf == 1:
        return qc_t.copy()
    qi = qc_t.inverse()
    folded = qc_t.copy()
    for _ in range((sf - 1) // 2):
        folded = folded.compose(qi).compose(qc_t)
    return folded

def zne_estimate(build_circuit_fn, params, H_den, sim_dm, cm, basis_gates, scale_factors=None) -> tuple:
    if scale_factors is None:
        scale_factors = [1, 3, 5]
    qc = build_circuit_fn(*params) if isinstance(params, tuple) else build_circuit_fn(params)
    qc_t = transpile(qc, coupling_map=cm, basis_gates=_BG_LIGHT, optimization_level=3)
    E_list = []
    for sf in scale_factors:
        qcf = fold_circuit(qc_t, sf)
        qcf.save_density_matrix()
        rho = np.array(sim_dm.run(qcf, shots=1).result().data(0)["density_matrix"])
        E_list.append(float(np.real(np.trace(H_den @ rho))))
    c = np.polyfit(scale_factors, E_list, deg=1)
    return float(np.polyval(c, 0.0)), 1, E_list

def cdr_train(build_circuit_fn, params_opt, H_den, sim_dm, sv_sim, cm, basis_gates, n_cdr=20, rng_seed=789) -> tuple:
    rng = np.random.default_rng(rng_seed)
    n1 = params_opt[0].size if isinstance(params_opt, tuple) else len(params_opt)
    n2 = params_opt[1].size if isinstance(params_opt, tuple) else 0
    X_noisy, X_zne, y = [], [], []
    for _ in range(n_cdr):
        if isinstance(params_opt, tuple):
            pert_l1 = rng.uniform(-0.5, 0.5, n1)
            pert_l2 = rng.uniform(-0.5, 0.5, n2)
            params_pert = (params_opt[0] + pert_l1, params_opt[1] + pert_l2, params_opt[2])
        else:
            params_pert = params_opt + rng.uniform(-0.5, 0.5, n1)
        E_ns = noisy_energy(build_circuit_fn, params_pert, H_den, sim_dm, cm, basis_gates)
        E_nl = noiseless_energy(build_circuit_fn, params_pert, H_den, sv_sim, cm, basis_gates)
        E_zne, _, _ = zne_estimate(build_circuit_fn, params_pert, H_den, sim_dm, cm, basis_gates)
        X_noisy.append(E_ns)
        X_zne.append(E_zne)
        y.append(E_nl)
    Xnoi_arr, Xzne_arr, y_arr = np.array(X_noisy), np.array(X_zne), np.array(y)

    def _fit(X_arr, y_arr):
        coeffs = np.polyfit(X_arr, y_arr, deg=1)
        y_pred = np.polyval(coeffs, X_arr)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        return coeffs, float(r2)

    c_noisy, r2_noisy = _fit(Xnoi_arr, y_arr)
    c_zne, r2_zne = _fit(Xzne_arr, y_arr)
    return (c_noisy, r2_noisy), (c_zne, r2_zne)

def cdr_apply(c: np.ndarray, E_noisy: float) -> float:
    return float(np.polyval(c, E_noisy))

def vd_estimate(build_circuit_fn, params, H_den, sim_dm, cm, basis_gates) -> tuple:
    qc = build_circuit_fn(*params) if isinstance(params, tuple) else build_circuit_fn(params)
    qc_t = transpile(qc, coupling_map=cm, basis_gates=_BG_LIGHT, optimization_level=3)
    qc_t.save_density_matrix()
    rho = np.array(sim_dm.run(qc_t, shots=1).result().data(0)["density_matrix"])
    rho2 = rho @ rho
    purity = float(np.real(np.trace(rho2)))
    E_vd = float(np.real(np.trace(H_den @ (rho2 / purity)))) if purity > 1e-10 else float(np.real(np.trace(H_den @ rho)))
    return E_vd, purity

# ── 7. IO helpers ──
def save_npz(data: dict, path: str, atomic: bool = True):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    safe = {}
    for k, v in data.items():
        if isinstance(v, (int, float, bool, str)): safe[k] = v
        elif isinstance(v, list):
            if v and isinstance(v[0], dict): safe[k] = json.dumps(v)
            else: safe[k] = np.array(v, dtype=object)
        else: safe[k] = v
    if atomic:
        fd, tmp = tempfile.mkstemp(dir=parent, suffix=".npz")
        os.close(fd)
        np.savez(tmp, **safe)
        os.replace(tmp, path)
    else:
        np.savez(path, **safe)

def merge_results(results_list: list[dict]) -> dict:
    merged = {}
    for key in results_list[0].keys():
        if key == "convergence": merged[key] = [r[key] for r in results_list]
        elif isinstance(results_list[0][key], (int, float, str, bool)):
            merged[key] = np.array([r[key] for r in results_list])
        else: merged[key] = [r[key] for r in results_list]
    return merged

# ── 8. Experiment Setup ──
def build_experiment_setup():
    if not _HAS_NATURE:
        raise ImportError("qiskit-nature is not installed; cannot build the H₄ Hamiltonian.")
    LOG("Building H₄ Hamiltonian...")
    h = build_h4_hamiltonian()
    H_den = h["HAMILTONIAN_DENSE"]
    e_nuc = h["E_NUC"]
    FCI = h["FCI_ENERGY"]
    hf_circuit = h["HF_CIRCUIT"]
    LOG(f"  FCI energy: {FCI:.6f} Ha, E_NUC: {e_nuc:.6f} Ha")

    LOG("Building FakeFez noise model...")
    nm, cm, bg, qpath = build_noise_model()
    LOG(f"  Quietest path: {qpath}")

    global _BG_LIGHT
    _BG_LIGHT = bg

    sim_dm = AerSimulator(method="density_matrix", noise_model=nm, coupling_map=cm)
    sv_sim = AerSimulator(method="statevector")
    LOG("  Simulators ready (DM + SV)")

    return {
        "H_den": H_den,
        "E_NUC": e_nuc,
        "FCI": FCI,
        "HF_CIRCUIT": hf_circuit,
        "sim_dm": sim_dm,
        "sv_sim": sv_sim,
        "cm": cm,
        "bg": bg,
        "qpath": qpath,
    }

def _count_gates(qc: QuantumCircuit, cm, basis_gates) -> int:
    qc_t = transpile(qc, coupling_map=cm, basis_gates=basis_gates, optimization_level=3)
    return qc_t.size()

# ── Phase 1: H4 VQE (captures layout A) ──
def run_h4_phase1(seed: int, setup: dict, n_outer: int = 35) -> dict:
    t_start = time.time()
    H = setup["H_den"]
    sim_dm = setup["sim_dm"]
    sv_sim = setup["sv_sim"]
    cm = setup["cm"]
    bg = setup["bg"]
    hf_circuit = setup["HF_CIRCUIT"]
    e_nuc = setup["E_NUC"]
    FCI = setup["FCI"]

    def build_fn(topaz, angles, active_pairs=None):
        return build_dual_lpte_circuit(topaz, angles, hf_circuit, active_pairs)

    rng = np.random.default_rng(seed)
    rho_l1 = rng.uniform(-2.0, 0.0, N_TOPAZ_LAYER // 2)
    rho_l2 = rng.uniform(-2.0, 0.0, N_TOPAZ_LAYER // 2)
    tau_l1 = rng.uniform(-2.0, 2.0, N_TOPAZ_LAYER // 2)
    tau_l2 = rng.uniform(-2.0, 2.0, N_TOPAZ_LAYER // 2)
    init_l1 = np.concatenate([rho_l1, tau_l1])
    init_l2 = np.concatenate([rho_l2, tau_l2])
    init_topaz = np.concatenate([init_l1, init_l2])

    init_angles = rng.uniform(-0.1, 0.1, N_ANGLES)

    gate_init = _count_gates(build_fn(init_topaz, init_angles), cm, bg)
    LOG(f"[Phase1 seed-{seed:02d}] H4 init gate count: {gate_init}")

    result = run_full_topaz_pipeline(
        build_fn, init_topaz, init_angles, H, sim_dm, cm, bg,
        n_outer=n_outer, seed=seed, e_nuc=e_nuc,
    )

    opt_topaz = result["best_topaz"]
    opt_angles = result["best_angles"]
    opt_energy = result["best_energy"]
    active_pairs = result.get("active_pairs", [True] * N_PAIRS)

    opt_params = (opt_topaz, opt_angles, active_pairs)
    noiseless = noiseless_energy(build_fn, opt_params, H, sv_sim, cm, bg)
    gate_final = _count_gates(build_fn(opt_topaz, opt_angles, active_pairs), cm, bg)

    E_noisy = opt_energy + e_nuc
    E_noiseless = noiseless + e_nuc
    LOG(f"[Phase1 seed-{seed:02d}] ├ E_noisy={E_noisy:.6f} Ha (ΔFCI={(E_noisy - FCI) * 1000:.1f} mH)")
    LOG(f"[Phase1 seed-{seed:02d}] ├ E_noiseless={E_noiseless:.6f} Ha (ΔFCI={(E_noiseless - FCI) * 1000:.1f} mH)")
    LOG(f"[Phase1 seed-{seed:02d}] ├ Gates: init={gate_init} final={gate_final}")

    return {
        "seed": seed,
        "topaz": opt_topaz,
        "angles": opt_angles,
        "active_pairs": active_pairs,
        "energy_noisy": E_noisy,
        "energy_noiseless": E_noiseless,
        "energy_fci": FCI,
        "delta_fci_noisy_mH": (E_noisy - FCI) * 1000,
        "delta_fci_noiseless_mH": (E_noiseless - FCI) * 1000,
        "gate_count_init": gate_init,
        "gate_count_final": gate_final,
        "duration_s": time.time() - t_start,
    }

# ── 8-qubit TFIM Hamiltonian ──
def build_tfim_hamiltonian(J: float = 1.0, h: float = 1.0) -> dict:
    from qiskit.quantum_info import SparsePauliOp
    n = N_QUBITS
    terms = []
    for i in range(n - 1):
        label = ["I"] * n
        label[i] = "Z"
        label[i + 1] = "Z"
        terms.append(("".join(label), -J))
    for i in range(n):
        label = ["I"] * n
        label[i] = "X"
        terms.append(("".join(label), -h))
    op = SparsePauliOp.from_list(terms)
    H_dense = op.to_matrix(sparse=True).toarray().astype(complex)
    fci = float(np.min(la.eigvalsh(H_dense)))
    LOG(f"[TFIM] J={J}, h={h}, FCI={fci:.6f} Ha (E_NUC=0)")
    return {
        "HAMILTONIAN_DENSE": H_dense,
        "E_NUC": 0.0,
        "FCI_ENERGY": fci,
    }

# ── TFIM circuit: same Ry+Rz/TOPAZ dual-layer structure, no HF reference ──
def build_tfim_circuit(topaz_params: np.ndarray, angle_params: np.ndarray = None,
                       active_pairs: list = None):
    qc = QuantumCircuit(N_QUBITS)
    for layer_idx in range(2):
        if angle_params is not None:
            off = layer_idx * N_ANGLES_PER_LAYER
            for q in range(N_QUBITS):
                qc.ry(angle_params[off + q], q)
                qc.rz(angle_params[off + N_QUBITS + q], q)
        layer_params = topaz_params[layer_idx * N_TOPAZ_LAYER : (layer_idx + 1) * N_TOPAZ_LAYER]
        qc.compose(build_lpte_layer(layer_params, layer_idx + 1, active_pairs), inplace=True)
    return qc

# ── Phase 2A: angle-only MMA (TOPAZ frozen at layout A) ──
def run_angle_only_mma(topaz: np.ndarray, init_angles: np.ndarray, build_fn,
                       H_den, sim_dm, cm, bg, n_inner: int = 30,
                       active_pairs=None, reg: float = 0.001) -> dict:
    angles = init_angles.copy()
    mma_ang = MMAOptimizer(N_ANGLES, move_limit=0.3)
    mma_ang.init(angles, delta=0.6)
    cur_loss = energy(topaz, angles, build_fn, H_den, sim_dm, cm, bg, active_pairs, reg)
    best_angles = angles.copy()
    best_loss = cur_loss
    prev_g_a = np.zeros(N_ANGLES)
    active_mask = np.ones(N_TOPAZ, dtype=bool)

    for inner_it in range(1, n_inner + 1):
        ag = angle_gradient(topaz, angles, build_fn, H_den, sim_dm, cm, bg,
                            active_pairs, active_mask, prev_g_a, reg)
        prev_g_a = ag.copy()
        new_angles = mma_ang.step(angles, ag)
        new_loss = energy(topaz, new_angles, build_fn, H_den, sim_dm, cm, bg, active_pairs, reg)
        if new_loss < cur_loss - 1e-8:
            angles = new_angles
            cur_loss = new_loss
            if cur_loss < best_loss:
                best_angles, best_loss = angles.copy(), cur_loss
        else:
            mma_ang.move_limit = max(0.02, mma_ang.move_limit * 0.8)
        mma_ang.update(angles, cur_loss)
        LOG(f"[2A] angle iter {inner_it}/{n_inner}: loss={cur_loss:.6f} Ha")

    return {"best_angles": best_angles, "best_energy": best_loss}

# ── Phase 2B: full pipeline from scratch (layout B) ──
def run_tfim_full(seed: int, build_fn, H_den, sim_dm, cm, bg, n_outer: int = 20) -> dict:
    rng = np.random.default_rng(seed)
    rho_l1 = rng.uniform(-2.0, 0.0, N_TOPAZ_LAYER // 2)
    rho_l2 = rng.uniform(-2.0, 0.0, N_TOPAZ_LAYER // 2)
    tau_l1 = rng.uniform(-2.0, 2.0, N_TOPAZ_LAYER // 2)
    tau_l2 = rng.uniform(-2.0, 2.0, N_TOPAZ_LAYER // 2)
    init_l1 = np.concatenate([rho_l1, tau_l1])
    init_l2 = np.concatenate([rho_l2, tau_l2])
    init_topaz = np.concatenate([init_l1, init_l2])
    init_angles = rng.uniform(-0.1, 0.1, N_ANGLES)

    result = run_full_topaz_pipeline(
        build_fn, init_topaz, init_angles, H_den, sim_dm, cm, bg,
        n_outer=n_outer, seed=seed,
    )
    active_pairs = result.get("active_pairs", [True] * N_PAIRS)
    result["active_pairs"] = active_pairs
    return result

def measure(build_fn, topaz, angles, active_pairs, H_den, sim_dm, sv_sim, cm, bg):
    params = (topaz, angles, active_pairs)
    E_noisy = noisy_energy(build_fn, params, H_den, sim_dm, cm, bg)
    E_noiseless = noiseless_energy(build_fn, params, H_den, sv_sim, cm, bg)
    gates = _count_gates(build_fn(topaz, angles, active_pairs), cm, bg)
    return E_noisy, E_noiseless, gates

# ── Orchestrator ──
def run_benchmark(seeds_h4: int = 1, seeds_tfim: int = 3,
                  n_outer_h4: int = 35, n_angle_inner: int = 30, n_outer_tfim: int = 20,
                  J: float = 1.0, h: float = 1.0, output_dir: str = "results_h4_tfim_transfer"):
    os.makedirs(output_dir, exist_ok=True)

    # ── Phase 1: H4 layout A ──
    LOG("=" * 70)
    LOG("PHASE 1: H4 LPTE VQE -> capture LAYOUT A")
    LOG("=" * 70)
    setup = build_experiment_setup()
    h4_runs = []
    for seed in range(seeds_h4):
        LOG(f"\n{'─'*40}\nH4 seed {seed:02d}/{seeds_h4-1}\n{'─'*40}")
        h4_runs.append(run_h4_phase1(seed, setup, n_outer=n_outer_h4))

    best_h4 = min(h4_runs, key=lambda r: r["delta_fci_noisy_mH"])
    layout_A_topaz = best_h4["topaz"].copy()
    layout_A_active = list(best_h4["active_pairs"])
    layout_A_gates = best_h4["gate_count_final"]
    LOG(f"LAYOUT A: H4 seed {best_h4['seed']:02d}, topaz={layout_A_topaz.size} params, "
        f"gates={layout_A_gates}, active_pairs={layout_A_active}")

    # ── Phase 2: TFIM ──
    LOG("\n" + "=" * 70)
    LOG("PHASE 2: 8-qubit TFIM transfer test")
    LOG("=" * 70)
    tfim = build_tfim_hamiltonian(J=J, h=h)
    H_t = tfim["HAMILTONIAN_DENSE"]
    FCI_t = tfim["FCI_ENERGY"]
    sim_dm = setup["sim_dm"]
    sv_sim = setup["sv_sim"]
    cm = setup["cm"]
    bg = setup["bg"]

    transfer_runs, full_runs = [], []
    for seed in range(seeds_tfim):
        rng = np.random.default_rng(seed + 1000)
        init_angles = rng.uniform(-0.1, 0.1, N_ANGLES)

        LOG(f"\n{'─'*40}\nTFIM seed {seed:02d}/{seeds_tfim-1}\n{'─'*40}")

        # 2A: transfer with frozen layout A TOPAZ
        LOG("[2A] Transfer: TOPAZ frozen at layout A, angle-only MMA...")
        t0 = time.time()
        res_a = run_angle_only_mma(layout_A_topaz, init_angles, build_tfim_circuit,
                                   H_t, sim_dm, cm, bg, n_inner=n_angle_inner,
                                   active_pairs=layout_A_active)
        E_noisy_a, E_nl_a, gates_a = measure(build_tfim_circuit,
                                             layout_A_topaz, res_a["best_angles"],
                                             layout_A_active, H_t, sim_dm, sv_sim, cm, bg)
        dur_a = time.time() - t0
        LOG(f"[2A] E_noisy={E_noisy_a:.6f} Ha (ΔFCI={(E_noisy_a - FCI_t) * 1000:.1f} mH), "
            f"E_noiseless={E_nl_a:.6f} Ha (ΔFCI={(E_nl_a - FCI_t) * 1000:.1f} mH), gates={gates_a}")

        # 2B: full from scratch -> layout B
        LOG("[2B] Baseline: full TOPAZ+angles pipeline from scratch...")
        t0 = time.time()
        res_b = run_tfim_full(seed, build_tfim_circuit, H_t, sim_dm, cm, bg, n_outer=n_outer_tfim)
        E_noisy_b, E_nl_b, gates_b = measure(build_tfim_circuit,
                                             res_b["best_topaz"], res_b["best_angles"],
                                             res_b["active_pairs"], H_t, sim_dm, sv_sim, cm, bg)
        dur_b = time.time() - t0
        LOG(f"[2B] E_noisy={E_noisy_b:.6f} Ha (ΔFCI={(E_noisy_b - FCI_t) * 1000:.1f} mH), "
            f"E_noiseless={E_nl_b:.6f} Ha (ΔFCI={(E_nl_b - FCI_t) * 1000:.1f} mH), gates={gates_b}")

        transfer_runs.append({
            "seed": seed, "energy_noisy": E_noisy_a, "energy_noiseless": E_nl_a,
            "energy_fci": FCI_t,
            "delta_fci_noisy_mH": (E_noisy_a - FCI_t) * 1000,
            "delta_fci_noiseless_mH": (E_nl_a - FCI_t) * 1000,
            "gate_count": gates_a, "duration_s": dur_a,
        })
        full_runs.append({
            "seed": seed, "energy_noisy": E_noisy_b, "energy_noiseless": E_nl_b,
            "energy_fci": FCI_t,
            "delta_fci_noisy_mH": (E_noisy_b - FCI_t) * 1000,
            "delta_fci_noiseless_mH": (E_nl_b - FCI_t) * 1000,
            "gate_count": gates_b, "duration_s": dur_b,
        })

        for tag, arr in (("2A", transfer_runs[-1]), ("2B", full_runs[-1])):
            save_npz(arr, os.path.join(output_dir, f"{tag}_seed_{seed:02d}.npz"))

    # ── Summary ──
    LOG("\n" + "=" * 70)
    LOG("TRANSFER TEST SUMMARY")
    LOG("=" * 70)
    LOG(f"  TFIM FCI = {FCI_t:.6f} Ha")
    dA = np.array([r["delta_fci_noisy_mH"] for r in transfer_runs])
    dB = np.array([r["delta_fci_noisy_mH"] for r in full_runs])
    dA_nl = np.array([r["delta_fci_noiseless_mH"] for r in transfer_runs])
    dB_nl = np.array([r["delta_fci_noiseless_mH"] for r in full_runs])
    LOG(f"  2A transfer (layout A frozen): noisy ΔFCI best={dA.min():.1f} worst={dA.max():.1f} avg={dA.mean():.1f} mH")
    LOG(f"     noiseless ΔFCI best={dA_nl.min():.1f} worst={dA_nl.max():.1f} avg={dA_nl.mean():.1f} mH")
    LOG(f"  2B full from scratch (layout B): noisy ΔFCI best={dB.min():.1f} worst={dB.max():.1f} avg={dB.mean():.1f} mH")
    LOG(f"     noiseless ΔFCI best={dB_nl.min():.1f} worst={dB_nl.max():.1f} avg={dB_nl.mean():.1f} mH")
    LOG(f"  Layout A gates: {layout_A_gates}, Layout B gates: {int(round(np.mean([r['gate_count'] for r in full_runs])))}")
    gain = (dB.mean() - dA.mean())
    LOG(f"  Transfer gain (avg ΔFCI full - avg ΔFCI transfer): {gain:+.1f} mH {'(transfer better)' if gain > 0 else '(full better)'}")
    LOG("=" * 70)

    return {"h4": h4_runs, "transfer": transfer_runs, "full": full_runs,
            "layout_A": {"topaz": layout_A_topaz, "active_pairs": layout_A_active,
                          "gates": layout_A_gates}}


if __name__ == "__main__":
    if not _HAS_NATURE:
        LOG("ERROR: qiskit-nature is not installed. Phase 1 (H4 LPTE) requires it. "
            "Install with `pip install qiskit-nature pyscf` or run TFIM-only. Exiting...")
        sys.exit(1)
    parser = argparse.ArgumentParser(description="H4-learned TOPAZ -> 8-qubit TFIM transfer test")
    parser.add_argument("--seeds-h4", type=int, default=1, help="Number of H4 phase-1 runs")
    parser.add_argument("--seeds-tfim", type=int, default=3, help="Number of TFIM transfer/full seeds")
    parser.add_argument("--n-outer-h4", type=int, default=35, help="H4 pipeline outer iterations")
    parser.add_argument("--n-angle-inner", type=int, default=30, help="Angle-only MMA iterations (2A)")
    parser.add_argument("--n-outer-tfim", type=int, default=20, help="TFIM full pipeline outer iterations (2B)")
    parser.add_argument("--J", type=float, default=1.0, help="TFIM coupling")
    parser.add_argument("--h", type=float, default=1.0, help="TFIM transverse field")
    parser.add_argument("--output", type=str, default="results_h4_tfim_transfer", help="Output directory")
    args = parser.parse_args()
    run_benchmark(seeds_h4=args.seeds_h4, seeds_tfim=args.seeds_tfim,
                  n_outer_h4=args.n_outer_h4, n_angle_inner=args.n_angle_inner,
                  n_outer_tfim=args.n_outer_tfim, J=args.J, h=args.h,
                  output_dir=args.output)
