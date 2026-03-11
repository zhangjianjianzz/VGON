import os
import warnings

import cvxpy as cp
import numpy as np
import torch
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer
from toqito.matrices import gen_gell_mann
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import normalize

warnings.filterwarnings('ignore')

device = torch.device("cpu")

SU9Basis = None
def __initialize_SU9Basis__():
    global SU9Basis
    if SU9Basis is None:
        n = 9
        k = 0
        SU9Basis = np.zeros((n**2, n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                SU9Basis[k, :, :] = gen_gell_mann(i, j, n)
                k = k + 1
        SU9Basis = torch.reshape(torch.from_numpy(SU9Basis).type(torch.complex64), (n**2, n**2, 1)).to(device)


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



def X2Paras(x_rec):
    # construct sigma

    theta = x_rec[:, 1:10]
    theta_norm = normalize(theta, p=2.0, dim=1)

    coeffs1 = [torch.pow(theta_norm[:,0],2).unsqueeze(-1).unsqueeze(-1),
               torch.pow(theta_norm[:,1],2).unsqueeze(-1).unsqueeze(-1),
               torch.pow(theta_norm[:,2],2).unsqueeze(-1).unsqueeze(-1),
               torch.pow(theta_norm[:,3],2).unsqueeze(-1).unsqueeze(-1),
               torch.pow(theta_norm[:,4],2).unsqueeze(-1).unsqueeze(-1),
               torch.pow(theta_norm[:,5],2).unsqueeze(-1).unsqueeze(-1),
               torch.pow(theta_norm[:,6],2).unsqueeze(-1).unsqueeze(-1),
               torch.pow(theta_norm[:,7],2).unsqueeze(-1).unsqueeze(-1),
               torch.pow(theta_norm[:,8],2).unsqueeze(-1).unsqueeze(-1)]

    sigma = None
    Sigma = 0
    for i in range(9):
        blank = torch.zeros((9,9),dtype=torch.complex64)
        blank[i,i] = 1
        if sigma is None:
            sigma = blank * coeffs1[i]
        else:
            sigma += blank * coeffs1[i]
        Sigma += coeffs1[i]

    # construct U
    __initialize_SU9Basis__()
    global SU9Basis
    L1 = torch.reshape(SU9Basis[1, :, :] * x_rec[:,10].unsqueeze(-1).unsqueeze(-1), (-1, 9, 9))

    for i in range(1, 80):
        L1 += torch.reshape(SU9Basis[i+1, :, :] * x_rec[:,10+i].unsqueeze(-1).unsqueeze(-1), (-1, 9, 9))

    U = torch.linalg.matrix_exp(-1j * L1)

    # construct state
    state = torch.bmm(U, sigma)
    state = torch.bmm(state, torch.conj(torch.transpose(U, 1, 2)))
    # print(state)

    # construct e1
    e1 = (torch.tanh(x_rec[:,0])+1) / 2
    return state, e1.type_as(state).unsqueeze(-1)


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
    __initialize_Ns__()

    global Ns
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
    problem = cp.Problem(cp.Minimize(cp.real(cp.trace(MU_locc @ rho))), constraints)
    return CvxpyLayer(problem, parameters=[rho, e1], variables=[e2_locc, P_locc, M0_locc, M1_locc])


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
    problem = cp.Problem(cp.Minimize(cp.real(cp.trace(MU_lo @ rho))), constraints)
    return CvxpyLayer(problem, parameters=[rho, e1], variables=[e2_lo, P_lo, M0_lo, M1_lo])


def Loss_Gap(x_rec):
    state, e1 = X2Paras(x_rec)
    cplayer_locc = E2Locc()
    cplayer_lo = E2Lo()
    e2_locc, _, _, _ = cplayer_locc(state, e1)
    e2_lo, _, _, _ = cplayer_lo(state, e1)
    gap = (e2_lo - e2_locc).type(torch.float32)
    return gap.sum()


class Model(nn.Module):
    def __init__(self, para_dim, z_dim, h_dim):
        super(Model, self).__init__()
        # encoder
        self.e1 = nn.ModuleList([nn.Linear(para_dim, h_dim[0],bias=True)])
        self.e1 += [nn.Linear(h_dim[i-1], h_dim[i],bias=True) for i in range(1, len(h_dim))]
        self.e2 = nn.Linear(h_dim[-1], z_dim,bias=True) # get mean prediction
        self.e3 = nn.Linear(h_dim[-1], z_dim,bias=True) # get mean prediction

        # decoder
        self.d4 = nn.ModuleList([nn.Linear(z_dim, h_dim[-1],bias=True)])
        self.d4 += [nn.Linear(h_dim[-i+1], h_dim[-i],bias=True) for i in range(2, len(h_dim)+1)]
        self.d5 = nn.Linear(h_dim[0], para_dim,bias=True)

    def encoder(self, x):
        h = F.relu(self.e1[0](x))
        for i in range(1, len(self.e1)):
            h = F.relu(self.e1[i](h))
        # get_mean
        mean = self.e2(h)
        # get_variance
        log_var = self.e3(h)  
        return mean, log_var

    def reparameterize(self, mean, log_var):
        eps = torch.randn(log_var.shape)
        std = torch.exp(log_var).pow(0.5) # square root
        z = mean + std * eps
        return z

    def decoder(self, z):
        out = F.relu(self.d4[0](z))
        for i in range(1, len(self.d4)):
            out = F.relu(self.d4[i](out))
        out = self.d5(out)
        return out

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        out = self.decoder(z)
        return out, mean, log_var
    
# Main Training Loop
if __name__ == '__main__':
    n_param, z_dim, batch_size = 90, 2, 3
    h_dim = [512, 256, 128]

    data_dist = dists.Uniform(0, 1)
    x = data_dist.sample([3000, n_param])
    train_loader = torch.utils.data.DataLoader(list(x), shuffle=True, batch_size=batch_size, drop_last=True)

    model = Model(n_param, z_dim, h_dim)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    for step, batch in enumerate(train_loader):
        batch = batch.to(device).view(-1, n_param)
        x_rec, mean, log_var = model(batch)
        rec_loss = Loss_Gap(x_rec)

        kl_div = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).mean() / batch_size
        loss = 1 - rec_loss + 0.1 * kl_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"Step {step}: Gap = {(rec_loss / batch_size).item():.6f}, KL = {kl_div.item():.6f}")

    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('results', 'gap.pth'))

    # Sample and save generated data
    z_samples = MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim)).sample([1000])
    generated_data = model.decoder(z_samples).detach().numpy()
    np.savetxt(os.path.join('results', 'generated_data.csv'), generated_data, delimiter=',')
