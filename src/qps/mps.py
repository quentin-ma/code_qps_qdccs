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
    """
    A quantum circuit simulator based on Matrix Product State data structure.
    """
    # Initiate Gamma_i with |0>
    def __init__(self, nqbits):
        self.dim = nqbits
        self.gammas = [np.array([1,0]).reshape((1,2,1)) for _ in range(self.dim)]
        pass

    """
    1-qubit function
    """
    def one_qubit_gate(self, gate, index):
        self.gammas[index] = np.einsum("ijk,jm->imk", self.gammas[index], gate)

    """
    2-qubit function
    """
    def two_qubits_gate(self, gate, index):
        # 1) einsum gamma[i], gamma[i+1] -> T
        # merge des deux tenseurs en un seul
        t = np.einsum("ijk,klm->ijlm", self.gammas[index], self.gammas[index + 1])
        # 1bis)
        t = t.reshape((t.shape[0], 4, t.shape[3]))
        # 2) einsum T gate
        # merge du tenseur de la porte au tenseur t
        t = np.einsum("ijk,jl->ilk", t, gate)
        # 3) SVD
        t = t.reshape((t.shape[0], 2, 2, t.shape[2]))
        # (A, 4, B) -> (A, 2, 2, B) 
        t = t.reshape((2 * t.shape[0], 2 * t.shape[3]))
        # (A, 4, B) -> (?, ?)
        u, s, v = np.linalg.svd(t)
        
        # remove nulls values
        non_zero_s = s.nonzero() # contient uniquement les indices non nuls de S
        
        u = u[:, : non_zero_s]
        u = u.reshape((u.shape[0], u.shape[2]))

        v = v[non_zero_s, :]
        s = s[non_zero_s]

        for i in range(s.shape[0]):
            u[:, i] *= s[i]
        
        u = u.reshape((self.gammas[index].shape[0], 2, s.shape[0]))
        v = v.reshape((s.shape[0], 2, self.gammas[index + 1].shape[2]))

        self.gammas[index] = u
        self.gammas[index + 1] = v

    def simulate_gate(self, gate, qubits):
        if len(qubits) == 1:
            self.one_qubit_gate(gate, qubits[0])
        elif len(qubits == 2) and (qubits[0] + 1) == qubits[1]:
            self.two_qubits_gate(gate, qubits[0])
        else:
            raise ValueError("Can't apply a gate on non consecutive qubits")

    def get_probability(self, classical_state):
        bra_0 = np.arrays([1,0])
        bra_1 = np.arrays([0,1])
        projectors = [bra_0 if p == '0' else bra_1 for p in classical_state]
        tensors = [np.einsum("abc,b->ac", self.gammas[i], projectors[i]) for i in range(self.dim)]

        contracted = np.array([1])
        for tensor in tensors:
            contracted = np.einsum("a,ab->b", contracted, tensor)
        amplitude = contracted[0]
        return abs(amplitude)**2

    def get_full_state(self):
        contracted = np.array([1])
        for tensor in self.tensors:
            contracted = np.einsum("la,abc->lbc", contracted, tensor)
            contracted = np.reshape(contracted.shape[0] * 2, contracted.shape[2])
        return contracted.reshape((1 << self.dim))
    
    def get_sample(self):
        res = []
        contracted = np.array([1])
        for i in range(self.dim):
            contracted = np.einsum("i,ikl->kl", contracted, self.gammas[i])
            bra_0 = np.array([1,0])
            contracted_projected = np.einsum("kl,k->l", contracted, bra_0)
            #  eq. to: contracted_projected = contracted[0, :]
            prob_0 = np.linalg.norm(contracted_projected) ** 2
            if np.random.random_sample() < prob_0:
                contracted = contracted[0, :] / np.linalg.norm(contracted[0, :])
                res.append(0)
            else:
                contracted = contracted_projected[1, :] / np.linalg.norm(contracted[1, :])
                res.append(1)
        return np.array(res)