from qps.mps import MPS
import numpy as np

mps = MPS(2)

CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

H = np.array([[1, 1], [1, -1]])/np.sqrt(2)
nbqubits = 300
mps.simulate_gate(H, [0])
mps.simulate_gate(CNOT, [0,1])

mps = MPS(nbqubits)
mps.simulate_gate(H, [0])
for i in range(N-1):
    mps.simulate_gate(CNOT, [i, i + 1])
#print(mps.get_probability("0" * nbqubits)) #simulation en 0(n)
for _ in range(10):
    mps.get_sample
