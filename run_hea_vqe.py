#!/usr/bin/env python3
"""HEA VQE benchmark (100% self‑contained).
Includes Hamiltonian setup, noise model, simulators, HEA circuit builder,
optimizer, full QEM stack (ZNE, CDR, VD) and result saving. No external custom package dependencies.
"""

import argparse
import os
import sys
import time
import warnings
import tempfile
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import networkx as nx

warnings.filterwarnings("ignore")

# ── Qiskit imports ──
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez

# ── Setup per‑script log file (clears any previous log) ──
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

# ── Base Constants ──
N_QUBITS = 4
N_PAIRS = 3

# ── 1. Hamiltonian Builder ──
_HAS_NATURE = False
try:
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.units import DistanceUnit
    _HAS_NATURE = True
except ImportError:
    pass

def build_h2_hamiltonian(bond_length: float = 0.735) -> dict:
    if not _HAS_NATURE:
        raise ImportError("qiskit-nature is required to build H₂ Hamiltonian.")
    atoms = f"H 0 0 0; H 0 0 {bond_length}"
    driver = PySCFDriver(atom=atoms, basis="sto3g", unit=DistanceUnit.ANGSTROM)
    problem = driver.run()
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())
    H_dense = qubit_op.to_matrix(sparse=True).toarray().astype(complex)
    E_nuc = problem.nuclear_repulsion_energy
    H_total = H_dense + E_nuc * np.eye(2**N_QUBITS, dtype=complex)
    fci_energy = float(np.min(la.eigvalsh(H_total)))
    pauli_operators = [(label, coeff) for label, coeff in qubit_op.to_list() if abs(coeff) > 1e-12]
    return {
        "HAMILTONIAN_DENSE": H_dense,
        "E_NUC": E_nuc,
        "FCI_ENERGY": fci_energy,
        "PAULI_OPERATORS": pauli_operators,
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
        if len(p) == 4:
            paths.append(p)
            return
        for nb in G.neighbors(n):
            if nb not in p:
                dfs(nb, p + [nb])
    for n in G.nodes:
        dfs(n, [n])

    scored = [(sum(G[p[i]][p[i+1]]['weight'] for i in range(3)), p) for p in paths]
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

# ── 3. HEA Circuit Builder ──
def build_hea(params: np.ndarray, depth: int = 3) -> QuantumCircuit:
    n_params = depth * (2 * N_QUBITS) + N_QUBITS
    if len(params) != n_params:
        raise ValueError(f"HEA params must have length {n_params}, got {len(params)}")
    qc = QuantumCircuit(N_QUBITS)
    
    # FIX: Initialize Hartree-Fock state |1100> for 2 electrons on 4 Jordan-Wigner qubits
    qc.x([0, 1])
    
    i = 0
    for _ in range(depth):
        for q in range(N_QUBITS):
            qc.ry(params[i], q); i += 1
        for q in range(N_QUBITS):
            qc.rz(params[i], q); i += 1
        for q in range(N_PAIRS):
            qc.cx(q, q + 1)
    for q in range(N_QUBITS):
        qc.ry(params[i], q); i += 1
    return qc

# ── 4. COBYLA Optimizer ──
def run_cobyla_vqe(build_fn, n_params, H_den, sim_dm, cm, bg, n_iters=500, init_params=None, noisy_energy_fn=None) -> dict:
    if init_params is None:
        rng = np.random.default_rng()
        init_params = rng.uniform(-np.pi, np.pi, n_params)
    energy_fn = noisy_energy_fn or (lambda x: noisy_energy(build_fn, x, H_den, sim_dm, cm, bg))
    log = []
    def callback(xk):
        e = energy_fn(xk)
        log.append({"iter": len(log), "energy": e, "time": time.time()})
    result = opt.minimize(
        energy_fn,
        init_params,
        method="COBYLA",
        options={"maxiter": n_iters, "rhobeg": 0.5, "tol": 1e-6},
        callback=callback,
    )
    return {
        "best_params": result.x,
        "best_energy": result.fun,
        "convergence": log,
        "n_iters": len(log),
        "success": result.success,
        "message": result.message,
    }

# ── 5. Energy & QEM Utils ──
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
    qc_t = transpile(qc, coupling_map=cm, basis_gates=basis_gates, optimization_level=3)
    E_list = []
    for sf in scale_factors:
        qcf = fold_circuit(qc_t, sf)
        qcf.save_density_matrix()
        rho = np.array(sim_dm.run(qcf, shots=1).result().data(0)["density_matrix"])
        E_list.append(float(np.real(np.trace(H_den @ rho))))
    deg = 2 if (len(E_list) >= 3 and E_list[0] > E_list[1] > E_list[2]) else 1
    c = np.polyfit(scale_factors, E_list, deg=deg)
    return float(np.polyval(c, 0.0)), deg, E_list

def cdr_train(build_circuit_fn, params_opt, H_den, sim_dm, sv_sim, cm, basis_gates, n_cdr=20, rng_seed=789) -> tuple:
    rng = np.random.default_rng(rng_seed)
    n1 = params_opt[0].size if isinstance(params_opt, tuple) else len(params_opt)
    n2 = params_opt[1].size if isinstance(params_opt, tuple) else 0
    X, y = [], []
    for _ in range(n_cdr):
        if isinstance(params_opt, tuple):
            pert_l1 = rng.uniform(-0.5, 0.5, n1)
            pert_l2 = rng.uniform(-0.5, 0.5, n2)
            params_pert = (params_opt[0] + pert_l1, params_opt[1] + pert_l2)
        else:
            params_pert = params_opt + rng.uniform(-0.5, 0.5, n1)
        E_ns = noisy_energy(build_circuit_fn, params_pert, H_den, sim_dm, cm, basis_gates)
        E_nl = noiseless_energy(build_circuit_fn, params_pert, H_den, sv_sim, cm, basis_gates)
        X.append(E_ns)
        y.append(E_nl)
    X_arr, y_arr = np.array(X), np.array(y)
    coeffs = np.polyfit(X_arr, y_arr, deg=1)
    y_pred = np.polyval(coeffs, X_arr)
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
    return coeffs, float(r2), X_arr, y_arr

def cdr_apply(c: np.ndarray, E_noisy: float) -> float:
    return float(np.polyval(c, E_noisy))

def vd_estimate(build_circuit_fn, params, H_den, sim_dm, cm, basis_gates) -> tuple:
    qc = build_circuit_fn(*params) if isinstance(params, tuple) else build_circuit_fn(params)
    qc_t = transpile(qc, coupling_map=cm, basis_gates=basis_gates, optimization_level=3)
    qc_t.save_density_matrix()
    rho = np.array(sim_dm.run(qc_t, shots=1).result().data(0)["density_matrix"])
    rho2 = rho @ rho
    purity = float(np.real(np.trace(rho2)))
    E_vd = float(np.real(np.trace(H_den @ (rho2 / purity)))) if purity > 1e-10 else float(np.real(np.trace(H_den @ rho)))
    return E_vd, purity

# ── 6. IO helpers ──
def save_npz(data: dict, path: str, atomic: bool = True):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    safe = {}
    for k, v in data.items():
        if isinstance(v, (int, float, bool, str)): safe[k] = v
        elif isinstance(v, list):
            if v and isinstance(v[0], dict): safe[k] = np.array([str(d) for d in v], dtype=object)
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

# ── 7. VQE Orchestrator functions ──
def build_experiment_setup():
    LOG("Building H₂ Hamiltonian...")
    h = build_h2_hamiltonian()
    H_den = h["HAMILTONIAN_DENSE"]
    E_NUC = h["E_NUC"]
    FCI = h["FCI_ENERGY"]
    pauli_ops = h["PAULI_OPERATORS"]
    LOG(f"  FCI energy: {FCI:.6f} Ha, E_NUC: {E_NUC:.6f} Ha, N_terms: {len(pauli_ops)}")

    LOG("Building FakeFez noise model...")
    nm, cm, bg, qpath = build_noise_model()
    LOG(f"  Quietest path: {qpath}")

    sim_dm = AerSimulator(method="density_matrix", noise_model=nm, coupling_map=cm)
    sv_sim = AerSimulator(method="statevector")
    LOG("  Simulators ready (DM + SV)")

    return {
        "H_den": H_den,
        "E_NUC": E_NUC,
        "FCI": FCI,
        "pauli_ops": pauli_ops,
        "sim_dm": sim_dm,
        "sv_sim": sv_sim,
        "cm": cm,
        "bg": bg,
        "qpath": qpath,
    }

def run_single_hea(seed: int, setup: dict, depth: int = 3, n_iters: int = 500):
    t_start = time.time()
    H = setup["H_den"]
    sim_dm = setup["sim_dm"]
    sv_sim = setup["sv_sim"]
    cm = setup["cm"]
    bg = setup["bg"]
    E_NUC = setup["E_NUC"]
    FCI = setup["FCI"]

    n_params = depth * (2 * N_QUBITS) + N_QUBITS
    LOG(f"[HEA seed-{seed:02d}] n_params={n_params}, depth={depth}, n_iters={n_iters}")

    build_fn = lambda p: build_hea(p, depth=depth)

    # HEA initialization
    rng = np.random.default_rng(seed)
    init_params = rng.uniform(-np.pi, np.pi, n_params)
    LOG(f"[HEA seed-{seed:02d}] Init params sample: {init_params[:3].round(3)}...")

    result = run_cobyla_vqe(
        build_fn, n_params, H, sim_dm, cm, bg,
        n_iters=n_iters, init_params=init_params,
    )

    opt_params = result["best_params"]
    opt_energy = result["best_energy"]
    LOG(f"[HEA seed-{seed:02d}] Raw noisy energy (electronic): {opt_energy:.6f} Ha")

    LOG(f"[HEA seed-{seed:02d}] Running ZNE...")
    zne_raw, zne_deg, _ = zne_estimate(build_fn, opt_params, H, sim_dm, cm, bg)

    LOG(f"[HEA seed-{seed:02d}] Running CDR (20 circuits)...")
    c_coeff, cdr_r2, _, _ = cdr_train(
        build_fn, opt_params, H, sim_dm, sv_sim, cm, bg,
        n_cdr=20, rng_seed=seed + 1000,
    )
    cdr_corrected = cdr_apply(c_coeff, opt_energy)
    zne_cdr = cdr_apply(c_coeff, zne_raw)

    LOG(f"[HEA seed-{seed:02d}] Running VD...")
    vd_raw, purity = vd_estimate(build_fn, opt_params, H, sim_dm, cm, bg)

    LOG(f"[HEA seed-{seed:02d}] Computing noiseless reference...")
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

    LOG(f"[HEA seed-{seed:02d}] ├ E_noisy={E_noisy:.6f} Ha (ΔFCI={d_noisy:.1f} mH)")
    LOG(f"[HEA seed-{seed:02d}] ├ E_noiseless={E_noiseless:.6f} Ha (ΔFCI={d_noiseless:.1f} mH)")
    LOG(f"[HEA seed-{seed:02d}] ├ E_ZNE={E_zne:.6f} Ha (ZNE degree={zne_deg})")
    LOG(f"[HEA seed-{seed:02d}] ├ E_CDR={E_cdr:.6f} Ha (R²={cdr_r2:.4f})")
    LOG(f"[HEA seed-{seed:02d}] ├ E_ZNE+CDR={E_zne_cdr:.6f} Ha")
    LOG(f"[HEA seed-{seed:02d}] ├ E_VD={E_vd:.6f} Ha (purity={purity:.4f})")
    LOG(f"[HEA seed-{seed:02d}] └ Duration: {duration:.1f}s")

    return {
        "seed": seed,
        "method": "hea",
        "success": result["success"],
        "n_optim_iters": result["n_iters"],
        "energy_noisy": E_noisy,
        "energy_noiseless": E_noiseless,
        "energy_fci": FCI,
        "energy_zne": E_zne,
        "energy_cdr": E_cdr,
        "energy_zne_cdr": E_zne_cdr,
        "energy_vd": E_vd,
        "purity": purity,
        "zne_fit_degree": zne_deg,
        "cdr_r2": cdr_r2,
        "delta_fci_noisy_mH": d_noisy,
        "delta_fci_noiseless_mH": d_noiseless,
        "convergence": result["convergence"],
        "duration_s": duration,
    }

def run_benchmark(n_seeds: int = 3, output_dir: str = "results_h2_vqe", depth: int = 3, n_iters: int = 500):
    os.makedirs(output_dir, exist_ok=True)
    setup = build_experiment_setup()
    method_results = []
    for seed in range(n_seeds):
        LOG(f"\n{'─'*40}\nSeed {seed:02d}/{n_seeds-1}\n{'─'*40}")
        result = run_single_hea(seed, setup, depth=depth, n_iters=n_iters)
        save_npz(result, os.path.join(output_dir, f"hea_seed_{seed:02d}.npz"))
        method_results.append(result)
    combined = merge_results(method_results)
    combined_path = os.path.join(output_dir, "hea_combined_results.npz")
    save_npz(combined, combined_path)
    LOG(f"\nCombined results → {combined_path}")

    LOG("\n" + "=" * 60)
    LOG("HEA BENCHMARK SUMMARY")
    LOG("=" * 60)
    FCI = method_results[0]["energy_fci"]
    LOG(f"  FCI = {FCI:.6f} Ha")
    for key, label in [("delta_fci_noisy_mH", "Noisy"),
                        ("delta_fci_noiseless_mH", "Noiseless"),
                        ("energy_zne", "ZNE"),
                        ("energy_cdr", "CDR"),
                        ("energy_zne_cdr", "ZNE+CDR"),
                        ("energy_vd", "VD")]:
        vals = np.array([r[key] for r in method_results])
        if "delta" in key:
            best = vals.min()
            worst = vals.max()
            avg = vals.mean()
            LOG(f"  {label}: best Δ={best:.2f} mH, worst Δ={worst:.2f} mH, avg Δ={avg:.2f} mH")
        else:
            deltas = (vals - FCI) * 1000
            best_d = deltas.min()
            worst_d = deltas.max()
            avg_d = deltas.mean()
            LOG(f"  {label}: best Δ={best_d:.2f} mH, worst Δ={worst_d:.2f} mH, avg Δ={avg_d:.2f} mH")

    noisy_deltas = np.array([r["delta_fci_noisy_mH"] for r in method_results])
    noiseless_deltas = np.array([r["delta_fci_noiseless_mH"] for r in method_results])
    best_i = noisy_deltas.argmin()
    worst_i = noisy_deltas.argmax()
    best_imp = (noisy_deltas[best_i] - noiseless_deltas[best_i]) / max(noisy_deltas[best_i], 1e-12) * 100
    worst_imp = (noisy_deltas[worst_i] - noiseless_deltas[worst_i]) / max(noisy_deltas[worst_i], 1e-12) * 100
    LOG(f"  Noiseless improvement: best={best_imp:.1f}%, worst={worst_imp:.1f}%")
    LOG(f"  Noisy energy: best seed={method_results[best_i]['energy_noisy']:.6f} Ha, "
        f"worst seed={method_results[worst_i]['energy_noisy']:.6f} Ha")
    LOG(f"  Best seed: {best_i}, Worst seed: {worst_i}")
    LOG("=" * 60)
    return method_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HEA VQE benchmark (100% self‑contained)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--depth", type=int, default=3, help="Circuit depth for HEA (default 3)")
    parser.add_argument("--n-iters", type=int, default=500, help="COBYLA iterations per seed")
    parser.add_argument("--output", type=str, default="results_h2_vqe", help="Output directory")
    args = parser.parse_args()
    run_benchmark(n_seeds=args.seeds, output_dir=args.output, depth=args.depth, n_iters=args.n_iters)
