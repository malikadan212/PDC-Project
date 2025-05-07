Community Detection-Based Parallel Quantum Circuit Simulator

A high-performance parallel quantum circuit simulator using tensor networks and community detection algorithms. This project leverages MPI for distributed computing and OpenCL for GPU-accelerated tensor operations, enabling efficient simulation of quantum circuits such as GHZ, QFT, and Random Quantum Circuits (RQC). It is inspired by the paper "A community detection-based parallel algorithm for quantum circuit simulation using tensor networks" by Pastor et al. (2025).
Overview
This simulator implements a community detection-based approach to parallelize the contraction of tensor networks representing quantum circuits. Using the Girvan-Newman algorithm, it partitions tensor networks into communities, contracts them in parallel with MPI, and accelerates tensor operations with OpenCL on GPUs. The project supports three circuit types: GHZ, QFT, and RQC, and includes an adaptive CPU/GPU contraction strategy for optimal performance.

Table of Contents

Installation
Usage
Features
Dependencies
Configuration
Examples
Contributing
License
Acknowledgements
Contact


Installation
Follow these steps to set up the project locally:

Clone the repository:
git clone https://github.com/yourusername/quantum-circuit-simulator.git
cd quantum-circuit-simulator


Set up a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Python dependencies:
pip install -r requirements.txt


Install MPI and OpenCL:

Ensure MPI (e.g., OpenMPI or MPICH) is installed on your system.
For GPU support, install OpenCL and ensure your system has an OpenCL-compatible GPU with appropriate drivers.




Usage
Run the simulator using the simulator.py script with command-line arguments to specify the circuit type, number of qubits, depth (for RQC), and seed (for RQC).
Command Syntax
python simulator.py --circuit <circuit_type> --qubits <num_qubits> --depth <circuit_depth> --seed <random_seed>

Arguments

--circuit: Type of quantum circuit (ghz, qft, rqc).
--qubits: Number of qubits in the circuit (integer).
--depth: Depth of the circuit (integer, required for rqc).
--seed: Random seed for generating RQC (integer, required for rqc).


Features

Supported Circuits: Simulates GHZ, QFT, and Random Quantum Circuits (RQC).
Community Detection: Uses the Girvan-Newman algorithm to partition tensor networks for efficient parallel processing.
Parallel Execution: Leverages MPI for distributed computing across multiple processes.
GPU Acceleration: Utilizes OpenCL for fast tensor contractions on GPUs.
Adaptive Contraction: Dynamically selects CPU or GPU for tensor contractions based on tensor size.


Dependencies
The following libraries and tools are required:

Python: 3.8 or higher
Python Libraries:
NumPy
NetworkX
Qiskit


System Requirements:
MPI (OpenMPI or MPICH)
OpenCL (for GPU acceleration)



Install Python dependencies with:
pip install -r requirements.txt


Configuration

OpenCL Device Selection: Set the OPENCL_DEVICE environment variable to specify the GPU device (if multiple GPUs are available):
export OPENCL_DEVICE=0  # Use the first GPU


MPI Configuration: Ensure your MPI environment is configured for distributed execution. Adjust settings based on your cluster or system setup.



Examples
Simulate a 10-Qubit GHZ Circuit
python simulator.py --circuit ghz --qubits 10

Expected Output: Logs the simulation process, including timings for circuit creation, tensor network building, contraction, and total time. Results are saved as visualizations (e.g., ghz_timings_new.png).
Simulate a 5-Qubit QFT Circuit
python simulator.py --circuit qft --qubits 5

Expected Output: Similar to the GHZ example, with results specific to the QFT circuit.
Simulate a 4-Qubit Random Quantum Circuit
python simulator.py --circuit rqc --qubits 4 --depth 3 --seed 42

Expected Output: Outputs results for the random circuit, varying based on the seed.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch:git checkout -b feature/your-feature-name


Commit your changes:git commit -m "Add your feature description"


Push to your fork:git push origin feature/your-feature-name


Open a pull request with a detailed description of your changes.

Please ensure your code follows the project's style guidelines and includes tests for new features.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements

Libraries and Tools: Thanks to the developers of NumPy, NetworkX, Qiskit, MPI, and OpenCL for their contributions to scientific computing.
Research Inspiration: This project builds on the work of Pastor et al. (2025) in their paper "A community detection-based parallel algorithm for quantum circuit simulation using tensor networks".


Contact
For questions, issues, or suggestions, reach out via:

GitHub Issues: Submit an Issue
Email: your.email@example.com

