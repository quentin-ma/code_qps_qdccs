"""
Implementation of a naive quantum circuit simulator
"""
from .simulator import StrongSimulator, WeakSimulator
import numpy as np

class Direct(StrongSimulator, WeakSimulator):
    """
    A naive quantum circuit simulator based on matrix-vector multiplication.
    """
    def __init__(self, nqbits):
        self.n = nqbits
        self.psi = np.zeros(pow(2, self.n))
        self.psi[0] = 1
        pass

    def simulate_gate(self, gate, qubits):
        new_shape = tuple ([2 for i in range(self.n)])

        k = len(qubits)
        
        self.psi = self.psi.reshape(new_shape)
        self.psi = np.moveaxis(self.psi, qubits, range(k))
        
        m_psi = np.reshape(self.psi, (pow(2, k), pow(2, self.n - k)))
        m_psi = gate.dot(m_psi)

        self.psi = np.reshape(m_psi, pow(2, self.n))

        self.psi = self.psi.reshape(new_shape)
        self.psi = np.moveaxis(self.psi, range(k), qubits)
        self.psi = self.psi.reshape((2 ** self.n))

    def get_probability(self, classical_state):
        return np.abs(self.psi[int(classical_state, 2)])**2

    def get_sample(self):
        idx = np.random.choice(range(1 << self.n), p=np.square(np.abs(self.psi)))
        return format(idx, "b")