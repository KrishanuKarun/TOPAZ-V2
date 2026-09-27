#!/usrApplicable/env python3
"""HVA VQE benchmark for H₂ (4-qubit, self-contained).
Includes Hamiltonian setup, noise model, simulators, HVA circuit builder,
optimizer, noisy/noiseless energy evaluation, and result saving.
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
import scipy.optimize as opt
import networkx as nx

warnings.filterwarnings("ignore")

# ── Qiskit imports ──
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez

# ── Setup per‑script log file ──
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
    pauli_operators = [(label, coeff) for label, coeff in qubit_op.to_list()
                       if abs(coeff) > 1e-12 and label != 'I' * N_QUBITS]
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

    scored = []
    for n in G.nodes:
        for n1 in G.neighbors(n):
            for n2 in G.neighbors(n1):
                if n2 == n: continue
                for n3 in G.neighbors(n2):
                    if n3 == n or n3 == n1: continue
                    p = [n, n1, n2, n3]
                    w = sum(G[p[i]][p[i+1]]['weight'] for i in range(3))
                    scored.append((w, p))
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

    # FIXED: Guard against KeyError by checking all qubits in quietest_path first
    for gn in bg:
        for qubits, error in fn._local_quantum_errors.get(gn, {}).items():
            if all(q in quietest_path for q in qubits):
                nq = tuple(p2v[q] for q in qubits)
                nm.add_quantum_error(error, gn, nq)

    return nm, cmap, bg, quietest_path

# ── 3. HVA Circuit Builder ──
def pauli_to_cx_evolution(pauli_label: str, theta: float) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)
    non_i = [q for q in range(N_QUBITS) if pauli_label[N_QUBITS - 1 - q] != 'I']
    if not non_i:
        return qc
    target = non_i[-1]
    for q in non_i:
        p = pauli_label[N_QUBITS - 1 - q]
        if p == 'X':
            qc.h(q)
        elif p == 'Y':
            qc.sdg(q); qc.h(q)
    for i in range(len(non_i) - 1):
        qc.cx(non_i[i], non_i[i + 1])
    qc.rz(2 * theta, target)
    for i in range(len(non_i) - 2, -1, -1):
        qc.cx(non_i[i], non_i[i + 1])
    for q in reversed(non_i):
        p = pauli_label[N_QUBITS - 1 - q]
        if p == 'Y':
            qc.h(q); qc.s(q)
        elif p == 'X':
            qc.h(q)
    return qc

def build_hva_h2(params: np.ndarray, pauli_operators: list, depth: int = 2) -> QuantumCircuit:
    n_terms = len(pauli_operators)
    qc = QuantumCircuit(N_QUBITS)
    for layer in range(depth):
        for term_idx, (pauli_label, _) in enumerate(pauli_operators):
            theta = params[layer * n_terms + term_idx]
            if all(c == 'I' for c in pauli_label):
                continue
            qc.compose(pauli_to_cx_evolution(pauli_label, theta), inplace=True)
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
    qc_t = transpile(qc, coupling_map=cm, basis_gates=basis_gates, optimization_level=1)
    qc_t.save_density_matrix()
    rho = np.array(sim_dm.run(qc_t, shots=1).result().data(0)["density_matrix"])
    return float(np.real(np.trace(H_den @ rho)))

def noiseless_energy(build_circuit_fn, params, H_den, sv_sim, cm, basis_gates) -> float:
    qc = build_circuit_fn(*params) if isinstance(params, tuple) else build_circuit_fn(params)
    qc_t = transpile(qc, basis_gates=basis_gates, optimization_level=0)
    qc_t.save_statevector()
    sv = np.array(sv_sim.run(qc_t, shots=1).result().data(0)["statevector"])
    return float(np.real(sv.conj() @ H_den @ sv))

# ── 6. IO helpers ──
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

def run_single_hva(seed: int, setup: dict, depth: int = 2, n_iters: int = 500):
    t_start = time.time()
    H = setup["H_den"]
    sim_dm = setup["sim_dm"]
    sv_sim = setup["sv_sim"]
    cm = setup["cm"]
    bg = setup["bg"]
    E_NUC = setup["E_NUC"]
    FCI = setup["FCI"]
    pauli_ops = setup["pauli_ops"]

    n_params = depth * len(pauli_ops)
    LOG(f"[HVA seed-{seed:02d}] n_params={n_params}, depth={depth}, n_iters={n_iters}")

    def build_fn(params):
        qc = QuantumCircuit(N_QUBITS)
        # FIXED: Initialize correct Hartree-Fock state |0101> (spatial 0_alpha, 0_beta)
        qc.x(0)
        qc.x(2)
        qc.compose(build_hva_h2(params, pauli_ops, depth=depth), inplace=True)
        return qc

    rng = np.random.default_rng(seed)
    init_params = rng.normal(0, 0.05, n_params)

    LOG(f"[HVA seed-{seed:02d}] Init params sample (normal 0,0.05): {init_params[:3].round(4)}...")

    result = run_cobyla_vqe(
        build_fn, n_params, H, sim_dm, cm, bg,
        n_iters=n_iters, init_params=init_params,
    )

    opt_params = result["best_params"]
    opt_energy = result["best_energy"]
    LOG(f"[HVA seed-{seed:02d}] Raw noisy energy (electronic): {opt_energy:.6f} Ha")

    LOG(f"[HVA seed-{seed:02d}] Computing noiseless reference...")
    noiseless = noiseless_energy(build_fn, opt_params, H, sv_sim, cm, bg)

    E_noisy = opt_energy + E_NUC
    E_noiseless = noiseless + E_NUC

    duration = time.time() - t_start
    d_noisy = (E_noisy - FCI) * 1000
    d_noiseless = (E_noiseless - FCI) * 1000

    LOG(f"[HVA seed-{seed:02d}] ├ E_noisy={E_noisy:.6f} Ha (ΔFCI={d_noisy:.1f} mH)")
    LOG(f"[HVA seed-{seed:02d}] ├ E_noiseless={E_noiseless:.6f} Ha (ΔFCI={d_noiseless:.1f} mH)")
    LOG(f"[HVA seed-{seed:02d}] └ Duration: {duration:.1f}s")

    return {
        "seed": seed,
        "method": "hva",
        "success": result["success"],
        "n_optim_iters": result["n_iters"],
        "energy_noisy": E_noisy,
        "energy_noiseless": E_noiseless,
        "energy_fci": FCI,
        "delta_fci_noisy_mH": d_noisy,
        "delta_fci_noiseless_mH": d_noiseless,
        "convergence": result["convergence"],
        "duration_s": duration,
    }

def run_benchmark(n_seeds: int = 3, output_dir: str = "results_h2_vqe", depth: int = 2, n_iters: int = 500):
    os.makedirs(output_dir, exist_ok=True)
    setup = build_experiment_setup()
    method_results = []
    for seed in range(n_seeds):
        LOG(f"\n{'─'*40}\nSeed {seed:02d}/{n_seeds-1}\n{'─'*40}")
        result = run_single_hva(seed, setup, depth=depth, n_iters=n_iters)
        save_npz(result, os.path.join(output_dir, f"hva_seed_{seed:02d}.npz"))
        method_results.append(result)
    combined = merge_results(method_results)
    combined_path = os.path.join(output_dir, "hva_combined_results.npz")
    save_npz(combined, combined_path)
    LOG(f"\nCombined results → {combined_path}")

    LOG("")
    LOG("=" * 50)
    LOG("HVA BENCHMARK SUMMARY")
    LOG("=" * 50)
    noisy_deltas = np.array([r["delta_fci_noisy_mH"] for r in method_results])
    noiseless_deltas = np.array([r["delta_fci_noiseless_mH"] for r in method_results])
    for label, vals in [("Noisy ΔFCI (mH)", noisy_deltas), ("Noiseless ΔFCI (mH)", noiseless_deltas)]:
        best = vals.min()
        worst = vals.max()
        avg = vals.mean()
        std = vals.std()
        LOG(f"  {label}: best={best:.2f}, worst={worst:.2f}, avg={avg:.2f} ± {std:.2f}")
    best_i = noisy_deltas.argmin()
    worst_i = noisy_deltas.argmax()
    best_imp = (noisy_deltas[best_i] - noiseless_deltas[best_i]) / max(noisy_deltas[best_i], 1e-12) * 100
    worst_imp = (noisy_deltas[worst_i] - noiseless_deltas[worst_i]) / max(noisy_deltas[worst_i], 1e-12) * 100
    LOG(f"  Noiseless improvement: best={best_imp:.1f}%, worst={worst_imp:.1f}%")
    LOG(f"  Best seed: {best_i}, Worst seed: {worst_i}")
    LOG("=" * 50)
    return method_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HVA VQE benchmark (100% self‑contained)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--depth", type=int, default=2, help="Circuit depth for HVA (default 2)")
    parser.add_argument("--n-iters", type=int, default=500, help="COBYLA iterations per seed")
    parser.add_argument("--output", type=str, default="results_h2_vqe", help="Output directory")
    args = parser.parse_args()
    run_benchmark(n_seeds=args.seeds, output_dir=args.output, depth=args.depth, n_iters=args.n_iters)