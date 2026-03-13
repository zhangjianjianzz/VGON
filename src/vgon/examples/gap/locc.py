import cvxpy as cp
import numpy as np

from .ns import __initialize_Ns__


def E2Locc():
    rho = cp.Parameter((9, 9))
    e1 = cp.Parameter(1)

    # parameters
    d = X = A = 3
    size = d**2

    # formulate sdp
    # variables
    P_locc = cp.Variable((2, X**2 * A**2), nonneg=True)  # [gamma, xyab]
    M0_locc = cp.Variable((size, size), hermitian=True)
    M1_locc = cp.Variable((size, size), hermitian=True)
    e2_locc = cp.Variable(1, nonneg=True)
    # construct MC, MU
    Ns = __initialize_Ns__()


    # reshape in cvxpy rearranges elements clolumn by clolumn, so transpose (.T) is needed
    PU_locc = cp.reshape(P_locc[0], (1, size**2))
    MU_locc = cp.reshape(PU_locc @ Ns, (size, size)).T
    PC_locc = cp.reshape(P_locc[1], (1, size**2))
    MC_locc = cp.reshape(PC_locc @ Ns, (size, size)).T

    # constraints
    constraints = [M0_locc >> 0, M1_locc >> 0]
    constraints += [
        e1 * np.eye(size, dtype=complex) - MC_locc == M0_locc + cp.partial_transpose(M1_locc, dims=(d, d), axis=1)
    ]
    # constraints of P - locc
    Pxya_b_locc = cp.reshape(cp.sum(P_locc, axis=0), (A, X**2 * A)).T
    constraints += [
        Pxya_b_locc[:, 0] - Pxya_b_locc[:, 1] == np.zeros(Pxya_b_locc.shape[0]),
        Pxya_b_locc[:, 0] - Pxya_b_locc[:, 2] == np.zeros(Pxya_b_locc.shape[0])
    ]
    P_xy_a_locc = cp.reshape(Pxya_b_locc[:, 0], (A, X**2)).T
    sum_matrix = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1]])
    P_x_a_locc = sum_matrix @ P_xy_a_locc
    constraints += [
        P_x_a_locc[:, 0] - P_x_a_locc[:, 1] == np.zeros(P_x_a_locc.shape[0]),
        P_x_a_locc[:, 0] - P_x_a_locc[:, 2] == np.zeros(P_x_a_locc.shape[0])
    ]
    constraints += [cp.sum(P_x_a_locc[:, 0]) == 1]
    constraints += [e2_locc == cp.real(cp.trace(MU_locc @ rho))]

    # problem
    parameter = [rho, e1]
    variable = [e2_locc, P_locc, M0_locc, M1_locc]
    return cp.Problem(cp.Minimize(cp.real(cp.trace(MU_locc @ rho))), constraints), parameter, variable


