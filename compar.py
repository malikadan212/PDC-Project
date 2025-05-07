#!/usr/bin/env python3

import sys
import time
import numpy as np
import networkx as nx
from mpi4py import MPI
import pyopencl as cl
from typing import List, Dict, Any, Tuple, Optional
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

# Define a Tensor as a NumPy array
Tensor = np.ndarray

# Configurable parameters
R4 = 16
MAT_DIM = R4 * R4
# Threshold for CPU vs GPU computation
GPU_SIZE_THRESHOLD = 64
USE_DYNAMIC_PADDING = True  # Enable smart padding

def create_qft_circuit_bis(n: int) -> QuantumCircuit:
    """Create a QFT circuit followed by two CNOT gates."""
    qft = QFT(n, do_swaps=False)
    qc = qft.compose(QuantumCircuit(n))
    if n >= 3:
        qc.cx(0, 1)
        qc.cx(1, 2)
    return qc

def create_ghz_circuit(n: int) -> QuantumCircuit:
    """Create a GHZ circuit with n qubits."""
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n-1):
        qc.cx(i, i+1)
    return qc

def create_rqc_circuit(n: int, depth: int, seed: int) -> QuantumCircuit:
    """Create a Random Quantum Circuit (RQC) with n qubits and specified depth."""
    import numpy.random as random
    random.seed(seed)
    qc = QuantumCircuit(n)
    single_qubit_gates = ['h', 'x', 'y', 'z', 't', 'tdg', 's', 'sdg']
    
    for d in range(depth):
        for i in range(n):
            gate = random.choice(single_qubit_gates)
            getattr(qc, gate)(i)
        for i in range(n-1):
            if random.random() > 0.5:
                qc.cx(i, i+1)
    
    for i in range(n):
        qc.h(i)
    return qc

def create_circuit(circuit_type: str, n: int, depth: int = 1, seed: int = 42) -> QuantumCircuit:
    """Factory method to create a circuit of specified type."""
    if circuit_type.lower() == "ghz":
        return create_ghz_circuit(n)
    elif circuit_type.lower() == "qft":
        return create_qft_circuit_bis(n)
    elif circuit_type.lower() == "rqc":
        return create_rqc_circuit(n, depth, seed)
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")

def gate_to_tensor(gate_name: str, params=None) -> Tensor:
    import numpy as np
    from math import cos, sin, pi, exp

    if params is None:
        params = []

    name = gate_name.lower()

    if name == "h":
        return np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
    elif name in ("cx", "cnot"):
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    elif name == "x":
        return np.array([[0,1],[1,0]], dtype=complex)
    elif name == "y":
        return np.array([[0,-1j],[1j,0]], dtype=complex)
    elif name == "z":
        return np.array([[1,0],[0,-1]], dtype=complex)
    elif name == "u1":
        [lam] = params
        return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=complex)
    elif name == "u2":
        [phi, lam] = params
        return np.array([[1, -np.exp(1j * lam)],[np.exp(1j * phi), np.exp(1j * (phi + lam))]], dtype=complex) / np.sqrt(2)
    elif name == "u3":
        [theta, phi, lam] = params
        return np.array([[np.cos(theta/2), -np.exp(1j * lam)*np.sin(theta/2)],[np.exp(1j * phi)*np.sin(theta/2), np.exp(1j * (phi + lam))*np.cos(theta/2)]], dtype=complex)
    elif name == "cp":
        [theta] = params
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,np.exp(1j * theta)]], dtype=complex)
    elif name == "cu1":
        [lam] = params
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,np.exp(1j * lam)]], dtype=complex)
    elif name == "p":
        [phi] = params
        return np.array([[1,0],[0,np.exp(1j * phi)]], dtype=complex)
    elif name == "swap":
        return np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    elif name == "rz":
        [phi] = params
        return np.array([[np.exp(-1j * phi/2),0],[0,np.exp(1j * phi/2)]], dtype=complex)
    elif name == "t":
        return np.array([[1, 0], [0, np.exp(1j * np.pi/4)]], dtype=complex)
    elif name == "tdg":
        return np.array([[1, 0], [0, np.exp(-1j * np.pi/4)]], dtype=complex)
    elif name == "s":
        return np.array([[1, 0], [0, np.exp(1j * np.pi/2)]], dtype=complex)
    elif name == "sdg":
        return np.array([[1, 0], [0, np.exp(-1j * np.pi/2)]], dtype=complex)
    else:
        raise ValueError(f"Unsupported gate: '{gate_name}' with params {params}")
        
def build_tensor_network(qc: QuantumCircuit) -> List[Dict[str, Any]]:
    """Convert a quantum circuit into a tensor network representation."""
    decomposed_circuit = qc.decompose().decompose()
    tensor_network = []
    for instr in decomposed_circuit.data:
        gate = instr.operation
        qubits = instr.qubits
        name = gate.name
        try:
            tensor = gate_to_tensor(name, list(gate.params) if hasattr(gate, 'params') else [])
            qubit_indices = [q._index for q in qubits]
            tensor_indices = [f"q{q}_in" for q in qubit_indices] + [f"q{q}_out" for q in qubit_indices]
            tensor_network.append({
                "gate_name": name,
                "tensor": tensor,
                "qubit_indices": qubit_indices,
                "tensor_indices": tensor_indices
            })
        except ValueError as e:
            print(f"Warning: {e}")
    return tensor_network

def build_line_graph(tn: List[Dict]) -> nx.Graph:
    """Build a line graph representing tensor connections."""
    G = nx.Graph()
    qubit_to_nodes = {}
    for i, node in enumerate(tn):
        G.add_node(i)
        for q in node["qubit_indices"]:
            qubit_to_nodes.setdefault(q, []).append(i)
    for nodes in qubit_to_nodes.values():
        for i in nodes:
            for j in nodes:
                if i < j:  # Avoid duplicate edges
                    G.add_edge(i, j)
    return G

def create_optimized_opencl_kernel():
    """Create a highly optimized OpenCL kernel for matrix multiplication."""
    return """
    // Most efficient matmul kernel with cache optimization
    __kernel void optimized_matmul(int M, int N, int K, 
                                  __global float* A, 
                                  __global float* B, 
                                  __global float* C) {
        // Work-group size
        const int TS = 16;  // Tile size - optimal for most GPUs
        
        // Get global position
        const int row = get_global_id(0);
        const int col = get_global_id(1);
        
        // Local position
        const int localRow = get_local_id(0);
        const int localCol = get_local_id(1);
        
        // Local memory for tiles with padding to avoid bank conflicts
        __local float Asub[16][17];  // Add padding column
        __local float Bsub[16][17];
        
        float sum = 0.0f;
        
        // Loop over tiles
        const int numTiles = (K + TS - 1) / TS;  // Ceiling division
        
        for (int t = 0; t < numTiles; t++) {
            // Collaborative loading with coalesced memory access
            const int tileRowA = row;
            const int tileColA = t * TS + localCol;
            
            if (tileRowA < M && tileColA < K) {
                Asub[localRow][localCol] = A[tileRowA * K + tileColA];
            } else {
                Asub[localRow][localCol] = 0.0f;
            }
            
            const int tileRowB = t * TS + localRow;
            const int tileColB = col;
            
            if (tileRowB < K && tileColB < N) {
                Bsub[localRow][localCol] = B[tileRowB * N + tileColB];
            } else {
                Bsub[localRow][localCol] = 0.0f;
            }
            
            // Sync to make sure data is loaded
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Compute on the tile with loop unrolling
            if (row < M && col < N) {
                // Manual loop unrolling for better performance
                #pragma unroll 4
                for (int k = 0; k < TS; k++) {
                    sum += Asub[localRow][k] * Bsub[k][localCol];
                }
            }
            
            // Sync before loading next tile
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        // Write result with bounds check
        if (row < M && col < N) {
            C[row * N + col] = sum;
        }
    }
    
    // Optimized complex matrix multiplication with vectorization
    __kernel void complex_matmul(int M, int N, int K,
                               __global float2* A,
                               __global float2* B,
                               __global float2* C) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        
        if (row < M && col < N) {
            float2 sum = (float2)(0.0f, 0.0f);
            
            // Loop with vectorization hints
            #pragma unroll 4
            for (int k = 0; k < K; k++) {
                float2 a = A[row*K + k];
                float2 b = B[k*N + col];
                
                // Complex multiplication using OpenCL built-in functions
                float2 prod;
                prod.x = a.x * b.x - a.y * b.y;
                prod.y = a.x * b.y + a.y * b.x;
                
                sum += prod;
            }
            
            C[row*N + col] = sum;
        }
    }
    """

def setup_opencl_context(rank: int, size: int) -> Tuple[Optional[cl.Context], Optional[cl.CommandQueue], Optional[cl.Kernel], Optional[Dict]]:
    """Optimized OpenCL setup with intelligent device selection and caching."""
    global _cl_cache
    
    try:
        if '_cl_cache' in globals() and rank in _cl_cache:
            return _cl_cache[rank]
        
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")
        
        sorted_platforms = sorted(platforms, key=lambda p: 
                                 -1 if "nvidia" in p.vendor.lower() else 
                                 (0 if "amd" in p.vendor.lower() else 1))
        
        ctx = None
        queue = None
        program = None
        
        devices_by_platform = []
        for platform in sorted_platforms:
            try:
                devices = platform.get_devices()
                if devices:
                    devices_by_platform.append((platform, devices))
            except:
                continue
        
        if not devices_by_platform:
            raise RuntimeError("No OpenCL devices found")
        
        all_devices = []
        for platform, devices in devices_by_platform:
            for device in devices:
                score = 1000 if device.type == cl.device_type.GPU else 0
                try:
                    score += device.global_mem_size / (1024 * 1024)
                except:
                    pass
                all_devices.append((score, platform, device))
        
        all_devices.sort(reverse=True)
        
        if all_devices:
            _, platform, device = all_devices[rank % len(all_devices)]
            
            try:
                ctx = cl.Context([device])
                queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
                program = cl.Program(ctx, create_optimized_opencl_kernel()).build(options="-cl-fast-relaxed-math")
                
                align_size = 256
                mat_size = ((MAT_DIM * MAT_DIM * np.float32().itemsize + align_size - 1) 
                           // align_size) * align_size
                
                mf = cl.mem_flags
                buffers = {
                    'A': cl.Buffer(ctx, mf.READ_ONLY, size=mat_size),
                    'B': cl.Buffer(ctx, mf.READ_ONLY, size=mat_size),
                    'C': cl.Buffer(ctx, mf.WRITE_ONLY, size=mat_size),
                    'C_result': np.zeros((MAT_DIM, MAT_DIM), dtype=np.float32)
                }
                
                result = (ctx, queue, program.optimized_matmul, buffers)
                
                if '_cl_cache' not in globals():
                    _cl_cache = {}
                _cl_cache[rank] = result
                
                return result
            except Exception as e:
                print(f"Warning: OpenCL setup error for device {device.name}: {e}")
        
        print(f"Rank {rank}: Using CPU fallback")
        return None, None, None, None
        
    except Exception as e:
        print(f"OpenCL disabled: {e}")
        return None, None, None, None

def prepare_complex_tensor(tensor: np.ndarray, target_dim: int = None) -> np.ndarray:
    """Prepare tensor with dynamic padding to nearest suitable size."""
    real_part = np.real(tensor).astype(np.float32)
    
    if not USE_DYNAMIC_PADDING:
        return real_part
    
    if target_dim is not None:
        orig_rows, orig_cols = real_part.shape
        padded = np.zeros((target_dim, target_dim), dtype=np.float32)
        padded[:orig_rows, :orig_cols] = real_part
        return padded
    
    def next_pow2(n): return 1 << (n-1).bit_length()
    
    orig_rows, orig_cols = real_part.shape
    new_rows = next_pow2(orig_rows)
    new_cols = next_pow2(orig_cols)
    
    padded = np.zeros((new_rows, new_cols), dtype=np.float32)
    padded[:orig_rows, :orig_cols] = real_part
    return padded

def optimized_workload_plan(G: nx.Graph, tn: List[Dict], procs: int) -> List[List[int]]:
    """Improved workload balancing with communication-aware cost model."""
    costs = {}
    gate_weights = {
        "cx": 4.0, "cnot": 4.0,
        "h": 1.0, "x": 1.0, "y": 1.0, "z": 1.0,
        "t": 1.2, "tdg": 1.2, "s": 1.1, "sdg": 1.1
    }
    
    for i, tensor_data in enumerate(tn):
        gate_type = tensor_data["gate_name"].lower()
        cost = np.prod(tensor_data["tensor"].shape) * gate_weights.get(gate_type, 1.0)
        comm_cost = len(list(G.neighbors(i))) * 0.5
        costs[i] = cost + comm_cost
    
    if len(G.nodes) > procs:
        try:
            from sklearn.cluster import SpectralClustering
            if len(G.nodes) > 10:
                adj_matrix = nx.to_numpy_array(G)
                clustering = SpectralClustering(n_clusters=procs, 
                                               affinity='precomputed',
                                               random_state=42).fit(adj_matrix)
                communities = [[] for _ in range(procs)]
                for i, label in enumerate(clustering.labels_):
                    communities[label].append(i)
                
                if any(len(c) < 1 for c in communities):
                    return balanced_bin_packing(costs, procs)
                return communities
        except (ImportError, Exception):
            pass
    
    return balanced_bin_packing(costs, procs)

def balanced_bin_packing(costs: Dict[int, float], procs: int) -> List[List[int]]:
    """Advanced bin packing algorithm for balanced load distribution."""
    sorted_nodes = sorted(costs.keys(), key=lambda x: costs[x], reverse=True)
    
    bins = [(0, i, []) for i in range(procs)]
    
    for node in sorted_nodes:
        min_idx = min(range(len(bins)), key=lambda i: bins[i][0])
        cost, proc_id, nodes = bins[min_idx]
        bins[min_idx] = (cost + costs[node], proc_id, nodes + [node])
    
    communities = [[] for _ in range(procs)]
    for _, proc_id, nodes in bins:
        communities[proc_id] = nodes
    
    return communities

def cpu_contract(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Optimized CPU matrix multiplication."""
    if min(A.shape[0], A.shape[1], B.shape[1]) > 256:
        try:
            return np.dot(A, B)
        except:
            pass
    
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), dtype=A.dtype)
    
    block_size = 64
    
    for i0 in range(0, M, block_size):
        i_end = min(i0 + block_size, M)
        for k0 in range(0, K, block_size):
            k_end = min(k0 + block_size, K)
            for j0 in range(0, N, block_size):
                j_end = min(j0 + block_size, N)
                
                for i in range(i0, i_end):
                    for k in range(k0, k_end):
                        a_val = A[i, k]
                        for j in range(j0, j_end):
                            C[i, j] += a_val * B[k, j]
    
    return C

def optimized_gpu_contract(queue, kernel, A: np.ndarray, B: np.ndarray, buffers: Dict) -> np.ndarray:
    """Perform efficient matrix multiplication on GPU using pre-allocated buffers."""
    cl.enqueue_copy(queue, buffers['A'], A)
    cl.enqueue_copy(queue, buffers['B'], B)
    
    M, K = A.shape
    K, N = B.shape
    global_size = (M, N)
    local_size = (16, 16)
    
    local_size = (min(local_size[0], M), min(local_size[1], N))
    
    kernel(queue, global_size, local_size, 
           np.int32(M), np.int32(N), np.int32(K),
           buffers['A'], buffers['B'], buffers['C'])
    
    cl.enqueue_copy(queue, buffers['C_result'], buffers['C'])
    
    return buffers['C_result'].copy()

def adaptive_contract(queue, kernel, A: np.ndarray, B: np.ndarray, buffers: Dict = None) -> np.ndarray:
    """Enhanced adaptive contraction with optimized CPU/GPU selection and operation batching."""
    if A.shape[0] < 32 or A.shape[1] < 32 or B.shape[1] < 32 or queue is None or kernel is None:
        return cpu_contract(A, B)
    
    try:
        if buffers is not None:
            A_buf = buffers['A']
            B_buf = buffers['B']
            C_buf = buffers['C']
            C = buffers['C_result']
            
            M, K = A.shape
            _, N = B.shape
            
            if M * K > MAT_DIM * MAT_DIM or K * N > MAT_DIM * MAT_DIM or M * N > MAT_DIM * MAT_DIM:
                return cpu_contract(A, B)
            
            cl.enqueue_copy(queue, A_buf, A, is_blocking=False)
            cl.enqueue_copy(queue, B_buf, B, is_blocking=False)
            
            local_size = (16, 16)
            if M * N < 4096:
                local_size = (8, 8)
            
            global_size = (
                ((M + local_size[0] - 1) // local_size[0]) * local_size[0],
                ((N + local_size[1] - 1) // local_size[1]) * local_size[1]
            )
            
            event = kernel(queue, global_size, local_size,
                       np.int32(M), np.int32(N), np.int32(K),
                       A_buf, B_buf, C_buf)
            
            cl.enqueue_copy(queue, C, C_buf, wait_for=[event])
            
            return C[:M, :N].copy()
        else:
            mf = cl.mem_flags
            A_buf = cl.Buffer(queue.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
            B_buf = cl.Buffer(queue.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
            C_buf = cl.Buffer(queue.context, mf.WRITE_ONLY, A.shape[0] * B.shape[1] * np.float32().itemsize)
            C = np.empty((A.shape[0], B.shape[1]), dtype=np.float32)
            
            M, K = A.shape
            _, N = B.shape
            
            local_size = (16, 16) if M * N > 4096 else (8, 8)
            global_size = (
                ((M + local_size[0] - 1) // local_size[0]) * local_size[0],
                ((N + local_size[1] - 1) // local_size[1]) * local_size[1]
            )
            
            event = kernel(queue, global_size, local_size,
                       np.int32(M), np.int32(N), np.int32(K),
                       A_buf, B_buf, C_buf)
            
            cl.enqueue_copy(queue, C, C_buf, wait_for=[event])
            
            A_buf.release()
            B_buf.release()
            C_buf.release()
            
            return C
            
    except cl.RuntimeError as e:
        print(f"OpenCL runtime error, falling back to CPU: {e}")
        return cpu_contract(A, B)

def main():
    """Optimized main function with improved communication pattern and asynchronous processing."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) < 3:
        if rank == 0:
            print("Usage: python script.py <circuit_type> <max_qubits> [depth] [seed]")
        sys.exit(1)

    circuit_type = sys.argv[1]
    max_qubits = int(sys.argv[2])
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    ctx, queue, kernel, buffers = setup_opencl_context(rank, size)

    if rank == 0:
        print(f"Circuit type: {circuit_type}")
        print(f"Max qubits: {max_qubits}")
        print(f"Depth: {depth}")
        print(f"Seed: {seed}")
        print(f"Number of processes: {size}")
        print(f"Matrix dimension: {MAT_DIM}x{MAT_DIM}")
        print("Qubits | Create(s) | BuildTN(s) | Contract(s) | Total(s)")
    
    for n in range(1, max_qubits + 1):
        if rank == 0:
            sys.stdout.flush()
            
        if rank == 0:
            circuit_start = time.time()
            qc = create_circuit(circuit_type, n, depth, seed)
            tn = build_tensor_network(qc)
            G = build_line_graph(tn)
            
            communities = optimized_workload_plan(G, tn, size)
            communities_tn_lists = [[tn[i] for i in comm] for comm in communities]
            circuit_time = time.time() - circuit_start
        else:
            communities_tn_lists = None
            circuit_time = 0

        local_tensors = comm.scatter(communities_tn_lists, root=0)
        
        comm.Barrier()
        contraction_start = time.time()
        
        if local_tensors:
            prepared_tensors = []
            for t in local_tensors:
                tensor = t["tensor"]
                if len(tensor.shape) == 2 and tensor.size == 4:
                    tensor = tensor.reshape((2, 2))
                elif len(tensor.shape) == 4 and tensor.size == 16:
                    tensor = tensor.reshape((4, 4))
                elif tensor.size == 16:
                    tensor = tensor.reshape((4, 4))
                
                prepared_tensors.append(prepare_complex_tensor(tensor, MAT_DIM))
            
            while len(prepared_tensors) > 1:
                smallest_cost = float('inf')
                smallest_idx = (0, 1)
                
                for i in range(len(prepared_tensors)):
                    for j in range(i+1, len(prepared_tensors)):
                        cost = prepared_tensors[i].size + prepared_tensors[j].size
                        if cost < smallest_cost:
                            smallest_cost = cost
                            smallest_idx = (i, j)
                
                i, j = smallest_idx
                a = prepared_tensors.pop(j)
                b = prepared_tensors.pop(i)
                result = adaptive_contract(queue, kernel, a, b, buffers)
                prepared_tensors.append(result)
            
            local_result = prepared_tensors[0] if prepared_tensors else np.eye(MAT_DIM, dtype=np.float32)
        else:
            local_result = np.eye(MAT_DIM, dtype=np.float32)
            
        contraction_time = time.time() - contraction_start
        
        reduction_start = time.time()
        final_result = None
        
        for d in range(int(np.log2(size)) + 1):
            step = 2**d
            if rank % (2*step) == 0 and rank + step < size:
                other_result = comm.recv(source=rank + step)
                if local_result is not None:
                    if other_result is not None:
                        local_result = adaptive_contract(queue, kernel, local_result, other_result, buffers)
            elif rank % (2*step) == step:
                comm.send(local_result, dest=rank - step)
                local_result = None
                
        reduction_time = time.time() - reduction_start	
            
        if rank == 0:
            total_time = circuit_time + contraction_time + reduction_time
            
            print(f"{n:>6} | {circuit_time:>8.6f} | {reduction_time:>10.6f} | {contraction_time:>11.6f} | {total_time:>12.6f}")
            
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                print(f"  Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
            except ImportError:
                pass
        
        comm.Barrier()
    
    if rank == 0:
        print("\nSimulation completed successfully with optimized performance!")

if _name_ == "_main_":
    main()
