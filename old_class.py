import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import newton_krylov, minimize

def expit(x):
    return 1 / (1 + np.exp(-x))

k_water = lambda x: 0.6*(x)**2
k_oil = lambda x: 0.2*(x-1)**2

# Сетка по x
class TwoPhase():
    def __init__(self,
                 mu_h,
                 mu_o,
                 glad,
                 G,
                 mu_water,
                 k,
                 Q,
                 m,
                 beta_r,
                 t_0,
                 T,
                 nt,
                 tt,
                 pow_n,
                 x_0,
                 L,
                 nx,
                 p_0,
                 s_0,
                 s_left,
                 s_right) -> None:
        self.mu_h = mu_h
        self.mu_o = mu_o
        self.glad = glad
        self.G = G
        self.mu_water = mu_water
        self.k = k
        self.Q = Q
        self.m = m
        self.beta_r = beta_r
        self.t_0 = t_0
        self.T = T
        self.nt = nt
        self.tt = tt
        self.pow_n = pow_n
        self.x_0 = x_0
        self.L = L
        self.nx = nx
        self.p_0 = p_0
        self.s_0 = s_0
        self.s_left = s_left
        self.s_right = s_right

        self.time = np.linspace(t_0, T, int(nt))
        self.dt = self.time[1]-self.time[0]
        self.tt = self.time.flat[np.abs(self.time - tt).argmin()]
        self.x = np.linspace(0, L, nx)
        self.dx = self.x[1]-self.x[0]

    def mu_oil(self, grad, mu_oil_x):
        grad = abs(grad)
        mu = (self.mu_h-self.mu_o) * expit (self.glad * (-grad + self.G)) + self.mu_o
        res = np.where(mu < mu_oil_x, mu, mu_oil_x)
        return res

    def rate(self, t):
        n = self.pow_n
        a = self.Q*(self.T-self.t_0) / (self.tt**(n+1)/(n+1) + self.tt**n*(self.T-self.tt))
        return np.where(t<self.tt,a*t**(n),a*self.tt**(n))

    def lam_w(self, s):
        return k_water(s) / self.mu_water

    def lam_o(self, s, grad, mu_oil_x):
        return k_oil(s) / self.mu_oil(grad, mu_oil_x)

    def beta(self, s, grad, mu_oil_x):
        return (self.lam_o(s, grad, mu_oil_x) + self.lam_w(s)) * self.k

    def solution_init(self):
        p = np.zeros((self.nt, self.nx))
        p[0, :] = self.p_0

        s = np.zeros((self.nt, self.nx))
        s[0,:] = self.s_0
        s[:,0] = self.s_left
        return p, s

    def residual(self, u_new, u_old, s_old, mu, t_step, dt, dx):
        """Выражение невязки для метода Ньютона-Крылова."""
        N = len(u_new)
        res = np.zeros(N)
        grad = np.gradient(u_new)
        k_left = (self.beta(s_old[:-2], grad[:-2], mu[:-2]) + self.beta(s_old[1:-1], grad[1:-1], mu[1:-1])) / 2
        k_right = (self.beta(s_old[1:-1], grad[1:-1], mu[1:-1]) + self.beta(s_old[2:], grad[2:], mu[2:])) / 2
        res[1:-1] = self.beta_r*(u_new[1:-1] - u_old[1:-1]) / dt - \
                    (k_right * (u_new[2:] - u_new[1:-1]) / dx**2 - k_left * (u_new[1:-1] - u_new[:-2]) / dx**2)

        # Граничные условия
        res[0] = (u_new[1] - u_new[0]) / dx * k_water(s_old[0]) / self.mu_water*self.k + self.rate(t_step)
        res[-1] = u_new[-1] - self.p_0

        return res

    # Метод Ньютона-Крылова для решения нелинейной системы
    def solve_nonlinear(self, u_old, s_old, mu, t_step, dt, dx):
        """Решение системы уравнений с использованием метода Ньютона-Крылова."""
        u_new_guess = np.copy(u_old)  # начальное предположение
        u_new = newton_krylov(lambda u_new: self.residual(u_new, u_old, s_old, mu, t_step, dt, dx), u_new_guess, f_tol=1e-6)
        return u_new

    # Параметры фракционного потока
    def fw(self, s, p, x):
        """ Функция фракционного потока воды """
        grad = np.gradient(p, x)

        return self.lam_w(s) * self.k * grad

    def solve(self):
        p, s = self.solution_init()
        mu_oil_x = np.zeros(self.nx)+self.mu_h

        # Решение системы с использованием Ньютона-Крылова
        for t, t_step in enumerate(self.time[:-1]):
            # решение уравнения для давления
            p_old = p[t, :].copy()

            if t!=0:
                s_old = s[t-1, :].copy()
            else:
                s_old = s[t, :].copy()

            p_new = self.solve_nonlinear(p_old, s_old, mu_oil_x, t_step, self.dt, self.dx)
            p[t+1, :] = p_new

            # решение уравнения для насыщенности
            fw_val = self.fw(s_old, p_new, self.x)

            s_new = np.zeros_like(s[t,:])
            s_new[1:] = (s_old[1:] + self.dt/(self.m*self.dx) * (fw_val[1:] - fw_val[:-1])) - s_old[1:]*self.beta_r/self.m*(p_new[1:]-p_old[1:])
            s_new[0] = self.s_left
            s[t+1,:] = s_new
            
            if t!=0:
                mu_oil_x = self.mu_oil(np.gradient(p_new, self.x), mu_oil_x)

        return p, s
