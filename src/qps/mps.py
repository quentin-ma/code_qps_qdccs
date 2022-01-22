"""
Implementation of a MPS-based quantum circuit simulator
"""

from .simulator import StrongSimulator, WeakSimulator
import numpy as np

"""
Recommended numpy functions:
reshape, einsum, svd
"""

class MPS(StrongSimulator, WeakSimulator):
    n = 0
    array_g = []
    """
    A quantum circuit simulator based on Matrix Product State data structure.
    """
    # Initiate Gamma_i with |0>
    def __init__(self, nqbits):
        self.n = nqbits
        self.array_g = [np.array([1,0]) for i in range(nqbits)]
        self.array_g = np.reshape(self.array_g, (1,2,1))
        pass

    """
    1-qubit function
    """
    def one_qubit_gate(self, gate, qubit):
        pass


"""
2-qubit function
"""

"""
Measures
"""