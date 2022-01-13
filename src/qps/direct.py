"""
Implementation of a naive quantum circuit simulator
"""
from .simulator import StrongSimulator, WeakSimulator
import numpy as np

class Direct(StrongSimulator, WeakSimulator):
    n = 0
    k = 0
    psi = []
    """
    A naive quantum circuit simulator based on matrix-vector multiplication.
    """
    def __init__(self, nqbits):
        self.n = nqbits
        self.psi = np.zeros(pow(2, self.n))
        pass

    def simulate_gate(self, gate, qubits):
        print("initial shape", self.psi.shape)
        k = len(qubits)
        M = np.reshape(self.psi, (pow(2, k), pow(2, self.n - k)))
        VM = gate.dot(M)
        print(gate, qubits)
        self.psi = VM.reshape(VM, pow(2, self.n))

    def get_probability(self, classical_state):
        return 0

    def get_sample(self):
        return 0