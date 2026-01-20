import numpy as np
import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import networkx as nx

class QuantumExplorer:
    def __init__(self):
        # Use AerSimulator for local high-speed simulation
        self.simulator = AerSimulator()

    def find_promising_paths(self, current_node, neighbors, goal_node, topology):
        if not neighbors: return {}
        
        num_moves = len(neighbors)
        # Determine the minimum qubits needed to represent neighbors
        num_qubits = max(1, int(math.ceil(math.log2(num_moves))))
        qc = QuantumCircuit(num_qubits)
        
        # 1. Initialization (Superposition)
        qc.h(range(num_qubits))
        
        # 2. Logic-Guided Oracle
        # Calculate shortest path weights to the goal for each neighbor
        distances = [nx.shortest_path_length(topology.G, n, goal_node, weight='weight') for n in neighbors]
        best_idx = int(np.argmin(distances))
        best_binary = format(best_idx, f'0{num_qubits}b')
        
        # Mark the 'best' candidate in the quantum state
        for i, bit in enumerate(reversed(best_binary)):
            if bit == '0': qc.x(i)
        
        if num_qubits > 1:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        else: 
            qc.z(0)
            
        for i, bit in enumerate(reversed(best_binary)):
            if bit == '0': qc.x(i)
            
        # 3. Diffuser (Amplitude Amplification)
        qc.h(range(num_qubits))
        qc.x(range(num_qubits))
        if num_qubits > 1:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        else: 
            qc.z(0)
        qc.x(range(num_qubits))
        qc.h(range(num_qubits))
        
        # 4. Measure and Execute
        qc.measure_all()
        # Optimization: Transpile once and reduce shots to 25 to prevent KeyboardInterrupt
        t_qc = transpile(qc, self.simulator)
        # shots=25 speeds up the run by 4x compared to your previous version
        result = self.simulator.run(t_qc, shots=25).result()
        counts = result.get_counts()
        
        probs = {}
        for i, node in enumerate(neighbors):
            # Format bits to match the circuit measurement order
            binary_str = format(i, f'0{num_qubits}b')
            # Sum up counts; qiskit results are in MSB-first order
            probs[node] = counts.get(binary_str, 0) / 25
            
        return probs