import numpy as np


Ns = None  # [xyab, N_xyab.reshape(-1)]


def __initialize_Ns__():
    global Ns
    if Ns is None:
        # parameters
        X = A = d = 3
        # define measurements
        N_xa = [[] for _ in range(X)]  # [x][a]
        # N_xa[0]
        vec = np.array([[1], [0], [0]], dtype=complex)
        N_xa[0].append(vec @ vec.conj().T)
        vec = np.array([[0], [1], [0]], dtype=complex)
        N_xa[0].append(vec @ vec.conj().T)
        vec = np.array([[0], [0], [1]], dtype=complex)
        N_xa[0].append(vec @ vec.conj().T)
        # N_xa[1]
        vec = np.array([[np.exp(1j * 2 * np.pi / 3)], [np.exp(-1j * 2 * np.pi / 3)], [1]], dtype=complex) / np.sqrt(3)
        N_xa[1].append(vec @ vec.conj().T)
        vec = np.array([[np.exp(-1j * 2 * np.pi / 3)], [np.exp(1j * 2 * np.pi / 3)], [1]], dtype=complex) / np.sqrt(3)
        N_xa[1].append(vec @ vec.conj().T)
        vec = np.array([[1], [1], [1]], dtype=complex) / np.sqrt(3)
        N_xa[1].append(vec @ vec.conj().T)
        # N_xa[2]
        vec = np.array([[1], [-1], [0]], dtype=complex) / (-np.sqrt(2))
        N_xa[2].append(vec @ vec.conj().T)
        vec = np.array([[0], [0], [1]], dtype=complex)
        N_xa[2].append(vec @ vec.conj().T)
        vec = np.array([[1], [1], [0]], dtype=complex) / (-np.sqrt(2))
        N_xa[2].append(vec @ vec.conj().T)

        # calculate final measurement matrix
        Ns = np.zeros((X**2 * A**2, d**4), dtype=complex)  # [xyab, N_xyab.reshape(-1)]
        for x in range(X):
            for y in range(X):
                for a in range(A):
                    for b in range(A):
                        Ns[x * (X * A**2) + y * A**2 + a * A + b, :] = np.kron(N_xa[x][a], N_xa[y][b]).reshape(-1)

    return Ns