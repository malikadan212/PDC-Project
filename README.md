Community Detection-Based Parallel Quantum Circuit Simulator
This project implements a parallel quantum circuit simulator using tensor networks and community detection algorithms. It leverages MPI for distributed computing and OpenCL for GPU-accelerated tensor operations, enabling efficient simulation of quantum circuits such as GHZ, QFT, and Random Quantum Circuits (RQC). The simulator is inspired by the paper "A community detection-based parallel algorithm for quantum circuit simulation using tensor networks" by Pastor et al. (2025).
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
To install and set up the project, follow these steps:

Clone the repository:
git clone https://github.com/yourusername/projectname.git


Navigate to the project directory:
cd projectname


Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt


Set up MPI and OpenCL:

Ensure youomat have MPI (e.g., OpenMPI or MPICH) and OpenCL installed on your system.
For GPU support, make sure your system has an OpenCL-compatible GPU and the necessary drivers installed.



Usage
To run the quantum circuit simulator, use the following command structure:
python simulator.py --circuit <circuit_type> --qubits <num_qubits> --depth <circuit_depth> --seed <random_seed>

Command-Line Arguments

--circuit: Type of quantum circuit to simulate. Options: ghz, qft, rqc.
--qubits: Number of qubits in the circuit.
--depth: Depth of the circuit (only applicable for rqc).
--seed: Random seed for generating random circuits (only applicable for rqc).

For example:

Simulate a 10-qubit GHZ circuit:
python simulator.py --circuit ghz --qubits 10


Simulate a 5-qubit QFT circuit:
python simulator.py --circuit qft --qubits 5


Simulate a 4-qubit random quantum circuit with depth 3 and seed 42:
python simulator.py --circuit rqc --qubits 4 --depth 3 --seed 42



Features

Circuit Types Supported: GHZ, QFT, and Random Quantum Circuits (RQC).
Community Detection: Uses the Girvan-Newman algorithm to partition tensor networks into communities for efficient parallel processing.
Parallelization: Utilizes MPI for distributed computing across multiple processes.
GPU Acceleration: Leverages OpenCL for GPU-accelerated tensor contractions, optimizing performance for large-scale simulations.
Adaptive Contraction: Dynamically chooses between CPU and GPU for tensor contractions based on tensor size to maximize efficiency.

Dependencies
The project requires the following libraries and tools:

Python 3.x
NumPy
NetworkX
Qiskit
MPI (OpenMPI or MPICH)
OpenCL
(Any other specific libraries used in the project)

You can install the Python dependencies using:
pip install -r requirements.txt

Configuration

OpenCL Device Selection: Set the OPENCL_DEVICE environment variable to specify which GPU device to use (if multiple are available).
MPI Configuration: Ensure your MPI environment is properly configured for distributed execution. You may need to adjust settings based on your cluster or system setup.

Examples
Here are some example commands and their expected outputs:

Simulate a 10-qubit GHZ circuit:
python simulator.py --circuit ghz --qubits 10


Expected output: Simulation results including contraction times and final tensor values.


Simulate a 5-qubit QFT circuit:
python simulator.py --circuit qft --qubits 5


Expected output: Similar to above, with results specific to the QFT circuit.


Simulate a 4-qubit random quantum circuit with depth 3:
python simulator.py --circuit rqc --qubits 4 --depth 3 --seed 42


Expected output: Results for the random circuit, which may vary based on the seed.



Contributing
We welcome contributions to improve the simulator! To contribute:

Fork the repository.
Create a new branch for your feature or bug fix.
Make your changes and ensure they follow the project's coding style.
Add tests for new features or fixes.
Submit a pull request with a clear description of your changes.

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgements

Libraries and Tools: We thank the developers of NumPy, NetworkX, Qiskit, MPI, and OpenCL for their invaluable contributions to scientific computing.
Research Inspiration: This project was inspired by the paper "A community detection-based parallel algorithm for quantum circuit simulation using tensor networks" by Pastor et al. (2025).

Contact
For questions, issues, or suggestions, please contact:

Your Name: your.email@example.com
GitHub Issues: Project Issues Page

