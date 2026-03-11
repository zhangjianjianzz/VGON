import cvxpy as cp
import numpy as np

from . import __initialize_Ns__


def E2Lo():
    rho = cp.Parameter((9, 9))
    e1 = cp.Parameter(1)

    # parameters
    d = X = A = 3
    size = d**2

    # formulate sdp
    # variables
    P_lo = cp.Variable((2, X**2 * A**2), nonneg=True)  # [gamma, xyab]
    M0_lo = cp.Variable((size, size), hermitian=True)
    M1_lo = cp.Variable((size, size), hermitian=True)
    e2_lo = cp.Variable(1, nonneg=True)
    # construct MC, MU
    __initialize_Ns__()
    global Ns

    # reshape in cvxpy rearranges elements column by column, so transpose (.T) is needed
    PU_lo = cp.reshape(P_lo[0], (1, size**2))
    MU_lo = cp.reshape(PU_lo @ Ns, (size, size)).T
    PC_lo = cp.reshape(P_lo[1], (1, size**2))
    MC_lo = cp.reshape(PC_lo @ Ns, (size, size)).T

    # constraints
    constraints = [M0_lo >> 0, M1_lo >> 0]
    constraints += [
        e1 * np.eye(size, dtype=complex) - MC_lo == M0_lo + cp.partial_transpose(M1_lo, dims=(d, d), axis=1)
    ]
    # constraints of P - lo
    Pxy_ab_lo = cp.reshape(cp.sum(P_lo, axis=0), (A**2, X**2)).T
    for i in range(Pxy_ab_lo.shape[1] - 1):
        for j in range(i + 1, Pxy_ab_lo.shape[1]):
            constraints += [Pxy_ab_lo[:, i] - Pxy_ab_lo[:, j] == np.zeros(Pxy_ab_lo.shape[0])]
    constraints += [cp.sum(Pxy_ab_lo[:, 0]) == 1]
    constraints += [e2_lo == cp.real(cp.trace(MU_lo @ rho))]

    # problem
    parameter = [rho, e1]
    variable = [e2_lo, P_lo, M0_lo, M1_lo]
    return cp.Problem(cp.Minimize(cp.real(cp.trace(MU_lo @ rho))), constraints), parameter, variable
