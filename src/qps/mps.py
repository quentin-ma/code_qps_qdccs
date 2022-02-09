"""
Implementation of a MPS-based quantum circuit simulator
"""

from .simulator import StrongSimulator, WeakSimulator
import numpy as np

SWAP = np.array([
    [1,0,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [0,0,0,1]
])

class MPS(StrongSimulator, WeakSimulator):
    """
    A quantum circuit simulator based on Matrix Product State data structure.
    """

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
        t = np.einsum("ijk,klm->ijlm", self.gammas[index], self.gammas[index + 1])
        t = t.reshape((t.shape[0], 4, t.shape[3]))
        t = np.einsum("ijk,lj->ilk", t, gate)
        t = t.reshape((t.shape[0], 2, 2, t.shape[2]))
        t = t.reshape((2 * t.shape[0], 2 * t.shape[3]))

        u, s, v = np.linalg.svd(t)
        
        non_zero_s = s.nonzero()
        
        u = u[:, non_zero_s]
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
        elif len(qubits) == 2:
            k = (qubits[1] - qubits[0]) - 1
            for i in range(k):
                self.two_qubits_gate(SWAP, i)
            self.two_qubits_gate(gate, qubits[1] - 1)
            for i in range(k - 2, qubits[0], -1):
                self.two_qubits_gate(SWAP, i)
        else:
            raise ValueError("error")

    def get_probability(self, classical_state):
        bra_0 = np.array([1,0])
        bra_1 = np.array([0,1])
        projectors = [bra_0 if p == "0" else bra_1 for p in classical_state]
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
            # bra_0 = np.array([1,0])
            # contracted_projected = np.einsum("kl,k->l", contracted, bra_0)
            #  eq. to: contracted_projected = contracted[0, :]
            prob_0 = np.linalg.norm(contracted[0, :]) ** 2
            if np.random.random_sample() < prob_0:
                contracted = contracted[0, :] / np.linalg.norm(contracted[0, :])
                res.append(0)
            else:
                contracted = contracted[1, :] / np.linalg.norm(contracted[1, :])
                res.append(1)
        return np.array2string(np.array(res))