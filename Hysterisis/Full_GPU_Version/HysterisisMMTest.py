import HysterisisMM
import numpy as np

W = np.ndarray(shape=(1,), dtype=float, order = 'C')
W = np.ascontiguousarray(W, dtype=np.float32)
W[0] = 2
print("W.shape: " + str(W.shape[0]))
print("W[0]: " + str(W[0]))
print("W: " + str(W))

V = np.ndarray(shape=(1,), dtype=float, order='C')
V = np.ascontiguousarray(V, dtype=np.float32)
V[0] = 0
print("V.shape " + str(V.shape[0]))
print("V[0] " + str(V[0]))
print("V: " + str(V))

ext_input = [0] * 30
ext_input.extend([1]*40)
ext_input.extend([0.5]*30)
E = np.ndarray(shape=(100,), dtype=float, buffer=np.asarray(ext_input), order='C')
E = np.ascontiguousarray(E, dtype=np.float32)
print("E.shape " + str(E.shape[0]))
#print("E[0] " + str(E[0]))
print("E: " + str(E))

timestep = .1

print("Timestep: " + str(timestep))

x = HysterisisMM.hysterisis(100, timestep, 4, .5, layers=V, weights=W, external=E)
print("Final Result: " + str(x))

