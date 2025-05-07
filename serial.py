#!/usr/bin/env python3
# Fully serial CPU-only tensor network simulator (no GPU/OpenCL dependencies)


import sys
import time
import platform
import numpy as np
import networkx as nx
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from typing import List, Dict, Any

# Define a Tensor as a NumPy array
Tensor = np.ndarray  # alias for clarity

# User parameters for CPU computation
R4 = 16
MAT_DIM = R4 * R4

def naive_matrix_mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Triple-loop matrix multiplication (O(n³) complexity)."""
    m, k = a.shape
    _, n = b.shape
    result = np.zeros((m, n), dtype=a.dtype)
    for i in range(m):
        for j in range(n):
            for l in range(k):
                result[i, j] += a[i, l] * b[l, j]
    return result

# Quantum circuit generators
def create_qft_circuit_bis(n: int) -> QuantumCircuit:
    qft = QFT(n, do_swaps=False)
    qc = qft.compose(QuantumCircuit(n))
    if n >= 3:
        qc.cx(0, 1)
        qc.cx(1, 2)
    return qc

def create_ghz_circuit(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n-1):
        qc.cx(i, i+1)
    return qc

def create_rqc_circuit(n: int, depth: int, seed: int) -> QuantumCircuit:
    import numpy.random as random
    random.seed(seed)
    qc = QuantumCircuit(n)
    single_qubit_gates = ['h','x','y','z','t','tdg','s','sdg']
    for _ in range(depth):
        for i in range(n):
            gate = random.choice(single_qubit_gates)
            if gate in ('tdg','sdg'):
                getattr(qc, gate)(i)
            else:
                getattr(qc, gate)(i)
        for i in range(n-1):
            if random.random()>0.5:
                qc.cx(i, i+1)
    for i in range(n): qc.h(i)
    return qc

# Gate-to-tensor mapping
def gate_to_tensor(gate_name: str, params=None) -> Tensor:
    import numpy as _np
    from math import pi
    if params is None: params = []
    name = gate_name.lower()
    if name == 'h': return _np.array([[1,1],[1,-1]],complex)/_np.sqrt(2)
    if name in ('cx','cnot'): return _np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],complex)
    if name == 'x': return _np.array([[0,1],[1,0]],complex)
    if name == 'y': return _np.array([[0,-1j],[1j,0]],complex)
    if name == 'z': return _np.array([[1,0],[0,-1]],complex)
    if name == 'u1': lam, = params; return _np.array([[1,0],[0,_np.exp(1j*lam)]],complex)
    if name == 'u2': phi, lam = params; return _np.array([[1,-_np.exp(1j*lam)],[_np.exp(1j*phi),_np.exp(1j*(phi+lam))]],complex)/_np.sqrt(2)
    if name == 'u3': theta, phi, lam = params; return _np.array([[_np.cos(theta/2), -_np.exp(1j*lam)*_np.sin(theta/2)],[_np.exp(1j*phi)*_np.sin(theta/2), _np.exp(1j*(phi+lam))*_np.cos(theta/2)]],complex)
    if name == 't': return _np.array([[1,0],[0,_np.exp(1j*pi/4)]],complex)
    if name == 'tdg': return _np.array([[1,0],[0,_np.exp(-1j*pi/4)]],complex)
    if name == 's': return _np.array([[1,0],[0,_np.exp(1j*pi/2)]],complex)
    if name == 'sdg': return _np.array([[1,0],[0,_np.exp(-1j*pi/2)]],complex)
    if name == 'swap': return _np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],complex)
    if name == 'rz': phi, = params; return _np.array([[ _np.exp(-1j*phi/2),0],[0,_np.exp(1j*phi/2)]],complex)
    if name in ('cp','cu1'):
        theta, = params
        mat = _np.eye(4, dtype=complex)
        mat[3,3] = _np.exp(1j*theta)
        return mat
    raise ValueError(f"Unsupported gate: {gate_name}")

# Build tensor network
def build_tensor_network(qc: QuantumCircuit) -> List[Dict[str, Any]]:
    decomposed = qc.decompose().decompose()
    tn = []
    for instr in decomposed.data:
        name = instr.operation.name
        try:
            tensor = gate_to_tensor(name, list(instr.operation.params))
        except ValueError:
            continue
        qubits = [q._index for q in instr.qubits]
        if len(qubits) == 1:
            idxs = [f"q{qubits[0]}_in", f"q{qubits[0]}_out"]
        else:
            idxs = [*(f"q{q}_in" for q in qubits), *(f"q{q}_out" for q in qubits)]
        tn.append({"gate_name": name, "tensor": tensor, "qubit_indices": qubits, "tensor_indices": idxs})
    return tn

# Line graph and community detection
def build_line_graph(tn):
    G = nx.Graph()
    qubit_map = {}
    for i, node in enumerate(tn):
        G.add_node(i)
        for q in node['qubit_indices']:
            qubit_map.setdefault(q, []).append(i)
    for nodes in qubit_map.values():
        for i in nodes:
            for j in nodes:
                if i < j: G.add_edge(i, j)
    return G

def greedy_community_detection(G):
    return list(range(len(G)))

# CPU contraction using numpy.dot
def prepare_for_cpu(tensor: np.ndarray) -> np.ndarray:
    flat_real = np.real(tensor).astype(np.float32).flatten()
    flat_imag = np.imag(tensor).astype(np.float32).flatten()
    inter = np.empty(flat_real.size + flat_imag.size, dtype=np.float32)
    inter[0::2] = flat_real; inter[1::2] = flat_imag
    size = MAT_DIM * MAT_DIM
    if inter.size < size:
        pad = np.zeros(size, dtype=np.float32)
        pad[:inter.size] = inter
        inter = pad
    else:
        inter = inter[:size]
    return inter.reshape(MAT_DIM, MAT_DIM)

def contract_tensors_cpu(tn, community):
    mats = [prepare_for_cpu(tn[i]['tensor']) for i in community]
    while len(mats) > 1:
        a = mats.pop(0)
        b = mats.pop(0)
        mats.append(naive_matrix_mult(a, b))  # Use naive multiplication
    return mats[0] if mats else np.eye(MAT_DIM, dtype=np.float32)

# Circuit selector and CLI parser
def select_circuit(circuit_type, n, depth=None, seed=None):
    if circuit_type == 'qft': return create_qft_circuit_bis(n)
    if circuit_type == 'ghz': return create_ghz_circuit(n)
    if circuit_type == 'rqc': return create_rqc_circuit(n, depth, seed)
    raise ValueError(f"Unknown circuit type: {circuit_type}")

def improved_command_line_parsing():
    import argparse
    p = argparse.ArgumentParser(description='Tensor Network CPU Simulator')
    p.add_argument('circuit_type', choices=['qft','ghz','rqc'], nargs='?', default='qft')
    p.add_argument('depth', type=int, nargs='?', default=None)
    p.add_argument('seed', type=int, nargs='?', default=None)
    args = p.parse_args()
    if args.circuit_type == 'rqc' and (args.depth is None or args.seed is None):
        print("[Error] depth and seed needed for RQC; using defaults 10,42")
        args.depth, args.seed = 10, 42
    return args

# Main
if __name__ == '__main__':
    # Environment info
    print("===== Environment Information =====")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"Python: {platform.python_version()} on {platform.system()} {platform.release()}")
    print(f"NumPy: {np.__version__}, NetworkX: {nx.__version__}, Qiskit: {qiskit.__version__}")
    print("===================================\n")

    # Parse CLI args
    args = improved_command_line_parsing()
    ctype, depth, seed = args.circuit_type, args.depth, args.seed

    print(f"Running serial benchmark for circuit '{ctype.upper()}' from 1 to 25 qubits")
    print("Qubits | Create(s) | BuildTN(s) | Contract(s) | Total(s)")
    print("-----------------------------------------------------")

    for n in range(1, 25):
        overall_start = time.perf_counter()
        t0 = time.perf_counter()
        qc = select_circuit(ctype, n, depth, seed)
        t1 = time.perf_counter()
        tn = build_tensor_network(qc)
        t2 = time.perf_counter()
        community = greedy_community_detection(build_line_graph(tn))
        t3 = time.perf_counter()
        final = contract_tensors_cpu(tn, community)
        t4 = time.perf_counter()
        create_time = t1 - t0
        build_time = t2 - t1
        contract_time = t4 - t3
        total_time = t4 - t3 + t2 - t0
        print(f"{n:>6} | {create_time:>8.6f} | {build_time:>10.6f} | {contract_time:>11.6f} | {total_time:>8.6f}")
