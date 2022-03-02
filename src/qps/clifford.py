import numpy as np


class Tableau:
    """
    A class representing the stabilizer group of a stabilizer quantum state.
    """

    def __init__(self, dim):
        self.dim = dim
        self.z = np.identity(dim, dtype=np.uint8)
        self.x = np.zeros((dim, dim), dtype=np.uint8)
        self.p = np.zeros(dim, dtype=np.uint8)

    def hadamard(self, qbit):
        """
        Conjugates the group by a H gate
        """
        copy = self.z[qbit].copy()
        self.z[qbit] = self.x[qbit].copy()
        self.x[qbit] = copy
        self.p = self.p ^ (self.z[qbit] & self.x[qbit])

    def phase(self, qbit):
        """
        Conjugates the group by a S gate
        """
        self.p = self.p ^ (self.z[qbit] & self.x[qbit])
        self.z[qbit] = self.z[qbit] ^ self.x[qbit]

    def cnot(self, control, target):
        """
        Conjugates the group by a CNOT gate
        """
        array = np.ones(self.dim, dtype=np.uint8)
        self.p = self.p ^ (self.x[control] & self.z[target] & (self.x[target] ^ self.z[control] ^ array))
        self.z[control] = self.z[control] ^ self.z[target]
        self.x[target] = self.x[target] ^ self.x[control]

    def measure(self, qbit):
        """
        Measures a qbit.
        """
        if any(self.x[qbit]):
            # get rid of ones (except for x[k,p])
            i = np.argwhere(self.x[qbit])[0] # index of the first column
            indexes = np.argwhere(self.x[qbit])
            
            for index in indexes[1:]:
                self.p[index] ^= np.matmul(self.x[:, index], self.z[:, index]) % 2
                self.z[:, index] ^= self.z[:, index]
                self.x[:, index] ^= self.x[:, index]

            self.z[:, i] = np.zeros((self.dim, 1))
            self.x[:, i] = np.zeros((self.dim, 1))

            self.z[qbit, i] = 1
            
            bit = np.random.randint(2) # randomness
            
            self.p[i] = bit
            
            return bit

    def get_circuit(self):
        """
        Generate a circuit that prepares the stabilizer state.
        """
        # TODO :)