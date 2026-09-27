#!/usr/bin/env python3
"""LPTE‑TOPAZ VQE benchmark for H₂ (4-qubit, 100% self-contained)."""

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
H2_CHAIN = "H 0 0 0; H 0 0 0.735"
N_QUBITS = 4
N_PAIRS = N_QUBITS - 1  # 3 pairs

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

# ── 1. Hamiltonian Builder ──
_HAS_NATURE = False
try:
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.circuit.library import HartreeFock
    from qiskit_nature.units import DistanceUnit
    _HAS_NATURE = True
except ImportError:
    pass

def build_h2_hamiltonian() -> dict:
    if not _HAS_NATURE:
        raise ImportError("qiskit-nature is required to build H₂ Hamiltonian.")
    driver = PySCFDriver(atom=H2_CHAIN, basis="sto3g", unit=DistanceUnit.ANGSTROM)
    problem = driver.run()
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())
    H_dense = qubit_op.to_matrix(sparse=True).toarray().astype(complex)
    E_nuc = problem.nuclear_repulsion_energy
    H_total = H_dense + E_nuc * np.eye(2**N_QUBITS, dtype=complex)
    fci_energy = float(np.min(la.eigvalsh(H_total)))
    
    # Use Qiskit's native HartreeFock with the correct mapper API
    hf_circuit = HartreeFock(num_spatial_orbitals=problem.num_spatial_orbitals, 
                             num_particles=problem.num_particles, 
                             qubit_mapper=mapper)
    
    return {
        "HAMILTONIAN_DENSE": H_dense,
        "E_NUC": E_nuc,
        "FCI_ENERGY": fci_energy,
        "HF_CIRCUIT": hf_circuit
    }

# ── 2. Noise Setup ──
def find_quietest_4q_path() -> list[int]:
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
    quietest_path = find_quietest_4q_path()
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
    # Per-pair normalization. Reshape to (N_PAIRS, 3) and normalize each row.
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

def build_dual_lpte_circuit(topaz_params: np.ndarray, angle_params: np.ndarray = None, hf_circuit: QuantumCircuit = None, active_pairs: list[bool] = None) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)
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

def run_mma_stage(build_fn, init_topaz, init_angles, H_den, sim_dm, cm, bg, n_outer, seed, active_pairs=None) -> dict:
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

        LOG(f"    Stage iter {outer_it}/{n_outer}: E_total={cur_loss + E_NUC:.6f} Ha [{time.time()-t_i:.1f}s]")

    return {
        "best_topaz": best_topaz,
        "best_angles": best_angles,
        "best_energy": best_loss,
        "success": stagnation <= 20,
    }

def run_full_topaz_pipeline(build_fn, init_topaz, init_angles, H_den, sim_dm, cm, bg, n_outer, seed):
    LOG("  Stage 1: Continuous Relaxation...")
    res1 = run_mma_stage(build_fn, init_topaz, init_angles, H_den, sim_dm, cm, bg, n_outer, seed, active_pairs=None)
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
    res3 = run_mma_stage(build_fn, res1["best_topaz"], res1["best_angles"], H_den, sim_dm, cm, bg, max(10, n_outer // 2), seed + 100, active_pairs=active_pairs)
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

# ── 8. VQE Orchestrator functions ──
def build_experiment_setup():
    LOG("Building H₂ Hamiltonian...")
    h = build_h2_hamiltonian()
    H_den = h["HAMILTONIAN_DENSE"]
    global E_NUC
    E_NUC = h["E_NUC"]
    FCI = h["FCI_ENERGY"]
    hf_circuit = h["HF_CIRCUIT"]
    LOG(f"  FCI energy: {FCI:.6f} Ha, E_NUC: {E_NUC:.6f} Ha")

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
        "E_NUC": E_NUC,
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

def run_single_lpte(seed: int, setup: dict, n_outer: int = 35):
    t_start = time.time()
    H = setup["H_den"]
    sim_dm = setup["sim_dm"]
    sv_sim = setup["sv_sim"]
    cm = setup["cm"]
    bg = setup["bg"]
    hf_circuit = setup["HF_CIRCUIT"]
    global E_NUC
    E_NUC = setup["E_NUC"]
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
    LOG(f"[LPTE seed-{seed:02d}] Init gate count: {gate_init}")

    result = run_full_topaz_pipeline(
        build_fn, init_topaz, init_angles, H, sim_dm, cm, bg,
        n_outer=n_outer, seed=seed,
    )

    opt_topaz = result["best_topaz"]
    opt_angles = result["best_angles"]
    opt_energy = result["best_energy"]
    active_pairs = result["active_pairs"]
    
    LOG(f"[LPTE seed-{seed:02d}] Raw noisy energy (electronic): {opt_energy:.6f} Ha")

    opt_params = (opt_topaz, opt_angles, active_pairs)
    LOG(f"[LPTE seed-{seed:02d}] Running ZNE...")
    zne_raw, zne_deg, _ = zne_estimate(build_fn, opt_params, H, sim_dm, cm, bg)

    LOG(f"[LPTE seed-{seed:02d}] Running CDR (20 circuits)...")
    cdr_noisy, cdr_zne_model = cdr_train(
        build_fn, opt_params, H, sim_dm, sv_sim, cm, bg,
        n_cdr=20, rng_seed=seed + 1000,
    )
    cdr_corrected = cdr_apply(cdr_noisy[0], opt_energy)
    zne_cdr = cdr_apply(cdr_zne_model[0], zne_raw)
    cdr_r2 = cdr_noisy[1]

    LOG(f"[LPTE seed-{seed:02d}] Running VD...")
    vd_raw, purity = vd_estimate(build_fn, opt_params, H, sim_dm, cm, bg)

    gate_final = _count_gates(build_fn(opt_topaz, opt_angles, active_pairs), cm, bg)
    gates_pruned = gate_init - gate_final

    LOG(f"[LPTE seed-{seed:02d}] Computing noiseless reference...")
    noiseless = noiseless_energy(build_fn, opt_params, H, sv_sim, cm, bg)

    E_noisy = opt_energy + E_NUC
    E_noiseless = noiseless + E_NUC
    E_zne = zne_raw + E_NUC
    E_cdr = cdr_corrected + E_NUC
    E_zne_cdr = zne_cdr + E_NUC
    E_vd = vd_raw + E_NUC

    duration = time.time() - t_start
    d_noisy = (E_noisy - FCI) * 1000
    d_noiseless = (E_noiseless - FCI) * 1000

    LOG(f"[LPTE seed-{seed:02d}] ├ E_noisy={E_noisy:.6f} Ha (ΔFCI={d_noisy:.1f} mH)")
    LOG(f"[LPTE seed-{seed:02d}] ├ E_noiseless={E_noiseless:.6f} Ha (ΔFCI={d_noiseless:.1f} mH)")
    LOG(f"[LPTE seed-{seed:02d}] ├ E_ZNE={E_zne:.6f} Ha (ZNE degree={zne_deg})")
    LOG(f"[LPTE seed-{seed:02d}] ├ E_CDR={E_cdr:.6f} Ha (R²={cdr_r2:.4f})")
    LOG(f"[LPTE seed-{seed:02d}] ├ E_ZNE+CDR={E_zne_cdr:.6f} Ha")
    LOG(f"[LPTE seed-{seed:02d}] ├ E_VD={E_vd:.6f} Ha (purity={purity:.4f})")
    LOG(f"[LPTE seed-{seed:02d}] ├ Gates: init={gate_init} final={gate_final} pruned={gates_pruned}")
    LOG(f"[LPTE seed-{seed:02d}] └ Duration: {duration:.1f}s")

    return {
        "seed": seed,
        "method": "lpte",
        "success": result["success"],
        "energy_noisy": E_noisy,
        "energy_noiseless": E_noiseless,
        "energy_fci": FCI,
        "energy_zne": E_zne,
        "energy_cdr": E_cdr,
        "energy_zne_cdr": E_zne_cdr,
        "energy_vd": E_vd,
        "purity": purity,
        "delta_fci_noisy_mH": d_noisy,
        "delta_fci_noiseless_mH": d_noiseless,
        "gate_count_init": gate_init,
        "gate_count_final": gate_final,
        "gates_pruned": gates_pruned,
        "duration_s": duration,
    }

def run_benchmark(n_seeds: int = 10, output_dir: str = "results_h2_vqe", n_outer: int = 35):
    os.makedirs(output_dir, exist_ok=True)
    setup = build_experiment_setup()
    method_results = []
    for seed in range(n_seeds):
        LOG(f"\n{'─'*40}\nSeed {seed:02d}/{n_seeds-1}\n{'─'*40}")
        result = run_single_lpte(seed, setup, n_outer=n_outer)
        save_npz(result, os.path.join(output_dir, f"lpte_seed_{seed:02d}.npz"))
        method_results.append(result)
    combined = merge_results(method_results)
    combined_path = os.path.join(output_dir, "lpte_combined_results.npz")
    save_npz(combined, combined_path)
    LOG(f"\nCombined results → {combined_path}")

    LOG("\n" + "=" * 60)
    LOG("BENCHMARK SUMMARY")
    LOG("=" * 60)
    FCI = method_results[0]["energy_fci"]
    LOG(f"  FCI = {FCI:.6f} Ha")
    for key, label in [("delta_fci_noisy_mH", "Noisy"),
                        ("delta_fci_noiseless_mH", "Noiseless"),
                        ("energy_zne_cdr", "ZNE+CDR")]:
        vals = np.array([r[key] for r in method_results])
        if "delta" in key:
            best = vals.min(); worst = vals.max(); avg = vals.mean()
            LOG(f"  {label}: best Δ={best:.2f} mH, worst Δ={worst:.2f} mH, avg Δ={avg:.2f} mH")
        else:
            deltas = (vals - FCI) * 1000
            best_d = deltas.min(); worst_d = deltas.max(); avg_d = deltas.mean()
            LOG(f"  {label}: best Δ={best_d:.2f} mH, worst Δ={worst_d:.2f} mH, avg Δ={avg_d:.2f} mH")

    noisy_deltas = np.array([r["delta_fci_noisy_mH"] for r in method_results])
    noiseless_deltas = np.array([r["delta_fci_noiseless_mH"] for r in method_results])
    best_i = noisy_deltas.argmin()
    worst_i = noisy_deltas.argmax()
    best_imp = (noisy_deltas[best_i] - noiseless_deltas[best_i]) / max(noisy_deltas[best_i], 1e-12) * 100
    worst_imp = (noisy_deltas[worst_i] - noiseless_deltas[worst_i]) / max(noisy_deltas[worst_i], 1e-12) * 100
    LOG(f"  Noiseless improvement: best={best_imp:.1f}%, worst={worst_imp:.1f}%")
    gate_inits = np.array([r["gate_count_init"] for r in method_results])
    gate_finals = np.array([r["gate_count_final"] for r in method_results])
    gate_pruned = np.array([r["gates_pruned"] for r in method_results])
    LOG(f"  Gates: init={int(round(gate_inits.mean()))}, "
        f"final={int(round(gate_finals.mean()))}, "
        f"pruned={int(round(gate_pruned.mean()))}")
    LOG("=" * 60)
    return method_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPTE‑TOPAZ VQE benchmark for H₂ (100% self‑contained)")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    parser.add_argument("--n-outer", type=int, default=35, help="Outer MMA (TOPAZ) iterations per seed")
    parser.add_argument("--output", type=str, default="results_h2_vqe", help="Output directory")
    args = parser.parse_args()
    run_benchmark(n_seeds=args.seeds, output_dir=args.output, n_outer=args.n_outer)