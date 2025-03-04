import numpy as np
from scipy.optimize import newton_krylov, fsolve
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

def estimate_damping_from_peak_difference(time, values, time_delta, interval_fraction=0.1):
    """
    Оценивает коэффициент затухания, используя разницу амплитуд между максимумами и минимумами.

    :param time: Массив времени
    :param values: Массив значений сигнала
    :param interval_fraction: Доля интервала для анализа начала и конца
    :return: Оценка коэффициента затухания alpha
    """
    t0, t1 = time_delta
    # Filter the time and values within the time interval
    mask = (time >= t0) & (time <= t1)
    time = time[mask]
    values = values[mask]

    n_points = int(len(time) * interval_fraction)

    #  Начало интервала
    start_values = values[:n_points]
    A_max_start = np.max(start_values)
    A_min_start = np.min(start_values)
    A_start = A_max_start - A_min_start
    t_max_start = time[np.argmax(start_values)]
    t_min_start = time[np.argmin(start_values)]

    # Конец интервала
    end_values = values[-n_points:]
    A_max_end = np.max(end_values)
    A_min_end = np.min(end_values)
    A_end = A_max_end - A_min_end
    t_max_end = time[-n_points:][np.argmax(end_values)]
    t_min_end = time[-n_points:][np.argmin(end_values)]

    # Временной промежуток
    delta_t = t_max_end - t_max_start

    # Оценка коэффициента затухания
    if A_end > 0 and A_start > 0:
        alpha = - (1 / delta_t) * np.log(A_end / A_start)
    else:
        alpha = np.nan  # В случае, если амплитуды некорректны
    

    # Визуализация сигнала и выбранных точек
    plt.figure(figsize=(12, 6))
    plt.plot(time, values, label='Сигнал')

    # Отметим выбранные амплитуды
    plt.scatter(t_max_start, A_max_start, color='green', label='Макс амплитуда (начало)', zorder=5)
    plt.scatter(t_min_start, A_min_start, color='lime', label='Мин амплитуда (начало)', zorder=5)
    plt.scatter(t_max_end, A_max_end, color='red', label='Макс амплитуда (конец)', zorder=5)
    plt.scatter(t_min_end, A_min_end, color='orange', label='Мин амплитуда (конец)', zorder=5)

    # Отметим границы анализируемых интервалов
    plt.axvline(time[n_points], color='gray', linestyle='--', label='Граница начала')
    plt.axvline(time[-n_points], color='gray', linestyle='--', label='Граница конца')

    plt.title(f'Коэффициент затухания: alpha = {alpha:.4f}')
    plt.xlabel('Время (s)')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Вывод значений амплитуд и коэффициента затухания
    print(f"Начальная амплитуда: {A_start:.4f} (Макс: {A_max_start:.4f}, Мин: {A_min_start:.4f})")
    print(f"Конечная амплитуда: {A_end:.4f} (Макс: {A_max_end:.4f}, Мин: {A_min_end:.4f})")
    print(f"Оцененный коэффициент затухания: {alpha:.4f}")

    return alpha, A_start, A_end

def analyze_oscillations(time_array, values_array, time_delta):
    # Extract the time interval
    t0, t1 = time_delta
    # Filter the time and values within the time interval
    mask = (time_array >= t0) & (time_array <= t1)
    time_filtered = time_array[mask]
    values_filtered = values_array[mask]
    
    # Fourier Transform
    N = len(time_filtered)
    dt = time_filtered[1] - time_filtered[0]  # assuming uniform time steps
    freqs = fftfreq(N, dt)
    fft_values = fft(values_filtered)
    
    # Find the frequency corresponding to the maximum amplitude
    magnitudes = np.abs(fft_values)
    peak_frequency = freqs[np.argmax(magnitudes[1:N//2])]  # exclude DC component (freq 0)
    peak_amplitude = np.max(magnitudes[1:N//2])

    return peak_frequency, peak_amplitude

def expit(x):
    try:
        return 1 / (1 + np.exp(-x))
    except:
        return 0.

def find_peaks_in_ts(time, values, height_factor=0.3, distance_fraction=0.001, prominence_factor=0.05):
    """
    Находит пики во временном ряде и возвращает глобальные индексы исходного массива.

    :param time: Массив времени
    :param values: Массив значений функции
    :param time_delta: Интервал времени (t0, t1)
    :param height_factor: Фактор от максимальной амплитуды для минимальной высоты пика
    :param distance_fraction: Фракция от длины интервала для минимального расстояния между пиками
    :param prominence_factor: Фактор от максимальной амплитуды для выделенности пиков
    :return: Глобальные индексы пиков, значения пиков
    """

    # Автоматический расчет порогов
    max_amplitude = np.max(values)
    height = height_factor * max_amplitude
    distance = int(distance_fraction * len(time))
    prominence = prominence_factor * max_amplitude

    # Нахождение пиков
    peaks, _ = find_peaks(values, height=height, distance=distance, prominence=prominence)

    return peaks, values[peaks]


k_water = lambda x: 0.6*(x)**2
k_oil = lambda x: 0.2*(x-1)**2

dk_water = lambda x: 1.2*x
dk_oil = lambda x: 0.4*(x-1)

class TwoPhase():
    def __init__(self,
                 mu_h,
                 mu_o,
                 glad,
                 G,
                 mu_h_w,
                 mu_h_w_b,
                 mu_o_w,
                 glad_w,
                 glad_w_b,
                 G_w,
                 G_w_b,
                 k,
                 m0,
                 beta_r,
                 rho_w0,
                 beta_w,
                 rho_o0,
                 beta_o,
                 t_0,
                 T,
                 nt,
                 tt,
                 pow_n,
                 alfa_k,
                 x_0,
                 L,
                 S,
                 nx,
                 p_0,
                 s_0,
                 s_left,
                 s_right,
                 mu_type_o,
                 mu_type_w,
                 p_left=None,
                 p_right=None,
                 Q=None) -> None:
        self.mu_h = mu_h
        self.mu_o = mu_o
        self.glad = glad
        self.G = G
        self.mu_h_w = mu_h_w
        self.mu_h_w_b = mu_h_w_b
        self.mu_o_w = mu_o_w
        self.glad_w = glad_w
        self.glad_w_b = glad_w_b
        self.G_w = G_w
        self.G_w_b = G_w_b
        self.k = k
        self.Q = Q
        self.m0 = m0
        self.beta_r = beta_r
        self.rho_w0 = rho_w0
        self.beta_w = beta_w
        self.rho_o0 = rho_o0
        self.beta_o = beta_o
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
        self.alfa_k = alfa_k
        self.time = np.linspace(t_0, T, int(nt), dtype=np.float64)/self.T
        self.dt = self.time[1]-self.time[0]
        self.tt = self.time.flat[np.abs(self.time - tt).argmin()]
        self.x = np.linspace(0, L, nx, dtype=np.float64)/self.L
        self.dx = self.x[1]-self.x[0]
        self.mu_oil_arr = []
        self.mu_water_arr = []
        self.grad_p = []
        self.grad_mu_o = []
        self.grad_mu_w = []
        self.mu_stop_o = np.zeros_like(self.x)
        self.mu_stop_w = np.zeros_like(self.x)
        if mu_type_o not in ["mu_2side", "mu_run", "mu"]:
            raise NameError
        self.mu_type_o = mu_type_o
        self.mu_type_w = mu_type_w
        self.p_left = p_left
        self.p_right = p_right
        self.S = S

    def m(self, p):
        return self.m0 + self.beta_r*(p*self.p_0-self.p_0)

    def rho_w(self, p):
        return self.rho_w0*(1+self.beta_w*(p*self.p_0-self.p_0))

    def rho_o(self, p):
        return self.rho_o0*(1+self.beta_o*(p*self.p_0-self.p_0))
    
    def mu_water(self, grad, grad_old, mu_stop_w):
        grad = abs(grad)*self.p_0/self.L
        grad_old = abs(grad_old)*self.p_0/self.L
        mu = (self.mu_h_w-self.mu_o_w) * expit (self.glad_w * (-grad + self.G_w)) + self.mu_o_w
        if self.mu_type_w=='mu_2side':
            mu_new = (self.mu_h_w_b-self.mu_o_w) * expit (self.glad_w_b * (-grad + self.G_w_b)) + self.mu_o_w
            res = np.where(mu<=self.mu_o_w, 1, 0)
            res = np.where(mu_stop_w+res>=1, 1, 0)
            result = np.where(res==1, mu_new, mu)
        elif self.mu_type_w=='mu_run':
            result = mu
            res = self.mu_stop_w
        elif self.mu_type_w=='mu_lin':
            result = np.ones_like(grad)*self.mu_o_w
            res = self.mu_stop_w
        else:
            res = np.where(grad>grad_old, grad, grad_old)
            result = (self.mu_h_w-self.mu_o_w) * expit (self.glad * (-res + self.G_w)) + self.mu_o_w
        return result, res

    def mu_oil(self, grad, grad_old, mu_stop_o):
        grad = abs(grad)*self.p_0/self.L
        grad_old = abs(grad_old)*self.p_0/self.L
        mu = (self.mu_h-self.mu_o) * expit (self.glad * (-grad + self.G)) + self.mu_o
        if self.mu_type_o=='mu_2side':
            mu_new = (self.mu_h/1. - self.mu_o) * expit (self.glad * (-grad + self.G/100)) + self.mu_o
            res = np.where(mu<=self.mu_o*1., 1, 0)
            res = np.where(mu_stop_o+res>=1, 1, 0)
            result = np.where(res==1, mu_new, mu)
        elif self.mu_type_o=='mu_run':
            result = mu
            res = None
        else:
            res = np.where(grad>grad_old, grad, grad_old)
            result = (self.mu_h-self.mu_o) * expit (self.glad * (-res + self.G)) + self.mu_o
        return result, res

    def rate(self, t):
        n = self.pow_n
        t = t*self.T
        tt = self.tt*self.T
        a = self.Q*(self.T-self.t_0) / (tt**(n+1)/(n+1) + tt**n*(self.T-tt))
        # if t<1.5:
        return np.where(t<tt,a*t**(n),a*tt**(n))
        # else:
        #     return -np.where(t<self.tt,a*t**(n),a*self.tt**(n))

    def lam_w(self, s, p, grad, grad_old, mu):
        return self.k*k_water(s)*self.rho_w(p) / self.mu_water(grad, grad_old, mu)[0]

    def lam_o(self, s, p, grad, grad_old, mu):
        return self.k*k_oil(s)*self.rho_o(p) / self.mu_oil(grad, grad_old, mu)[0]

    def coef_dp_dt(self, p, s):
        return self.beta_r + self.m(p)*s*self.beta_w*self.rho_w0/self.rho_w(p)+\
                   self.m(p)*(1-s)*self.beta_o*self.rho_o0/self.rho_o(p)

    def s_function(self, t):
        """
        Функция, изменяющая s по линейному закону до t=tt, 
        после чего s фиксируется на уровне s1.

        :param t: текущее время
        :param tt: пороговое время
        :param s0: начальное значение s
        :param s1: конечное значение s
        :return: значение s в момент времени t
        """
        if t < self.tt:
            return self.s_0 + (self.s_left - self.s_0) * (t / self.tt)
        else:
            return self.s_left

    def solution_init(self):
        p = np.zeros((self.nt, self.nx), dtype=np.float64)
        p[0, :] = self.p_0/self.p_0

        s = np.zeros((self.nt, self.nx), dtype=np.float64)
        s[0,:] = self.s_0
        s[:,0] = [self.s_function(i) for i in self.time]
        return p, s

    def grad(self, p, dx, axis='center'):
        res = np.ones_like(p)
        if axis=='left':
            res[1:] = (p[1:]-p[:-1])/dx
            res[0] = (p[1]-p[0])/dx
        elif axis=='right':
            res[:-1] = (p[1:]-p[:-1])/dx
            res[-1] = (p[-1]-p[-2])/dx
        else:
            res = np.gradient(p, dx, edge_order=1)
        return res

    def residual(self, u_new, u_old, s_old, t_step, dt, dx):
        """Выражение невязки для метода Ньютона-Крылова."""
        N = len(u_new)
        res = np.zeros(N, dtype=np.float64)
        grad = self.grad(u_new, self.dx)
        grad_old = self.grad_p[-1]
        k_left_w = self.lam_w(s_old[1:-1], u_new[1:-1], grad[1:-1], grad_old[1:-1], self.mu_stop_w[1:-1])
        k_right_w = self.lam_w(s_old[2:], u_new[2:], grad[2:], grad_old[2:], self.mu_stop_w[2:])

        k_left_o = self.lam_o(s_old[1:-1], u_new[1:-1], grad[1:-1], grad_old[1:-1], self.mu_stop_o[1:-1])
        k_right_o = self.lam_o(s_old[2:], u_new[2:], grad[2:], grad_old[2:], self.mu_stop_o[2:])


        res[1:-1] = 1*self.coef_dp_dt(u_new[1:-1], s_old[1:-1])*(u_new[1:-1] - u_old[1:-1]) / dt - \
                    self.T/(self.rho_w(u_new[1:-1])*self.L**2)*(k_right_w*(u_new[2:] - u_new[1:-1]) / dx**2 - k_left_w*(u_new[1:-1] - u_new[:-2]) / dx**2) -\
                    self.T/(self.rho_o(u_new[1:-1])*self.L**2)*(k_right_o*(u_new[2:] - u_new[1:-1]) / dx**2 - k_left_o*(u_new[1:-1] - u_new[:-2]) / dx**2)

        # Граничные условия
        if self.p_left is not None:
            res[0] = u_new[0] - self.p_left/self.p_0
        else:
            # res[0] = -grad[0] + 8e+6/self.p_0*self.L
            res[0] = self.p_0/self.L*grad[0] * k_water(s_old[0])* self.k / self.mu_water(grad[0], grad_old[0], self.mu_stop_w[0])[0]*self.S + self.rate(t_step)
        res[-1] = u_new[-2] - self.p_right/self.p_0

        return res

    # Метод Ньютона-Крылова для решения нелинейной системы
    def solve_nonlinear(self, u_old, s_old, t_step, dt, dx):
        """Решение системы уравнений с использованием метода Ньютона-Крылова."""
        u_new_guess = np.copy(u_old)  # начальное предположение
        u_new = newton_krylov(lambda u_new: self.residual(u_new, u_old, s_old, t_step, dt, dx), u_new_guess)
        # u_new = least_squares(lambda u_new: self.residual(u_new, u_old, s_old, t_step, dt, dx), u_new_guess, ).x
        # u_new = root(lambda u_new: self.residual(u_new, u_old, s_old, t_step, dt, dx), u_new_guess, method='krylov').x
        return u_new

    # Параметры фракционного потока
    def fw(self, s, p, grad, grad_old, mu_w):
        """ Функция фракционного потока воды """
        grad = self.grad(p, self.dx)

        return self.lam_w(s, p, grad, grad_old, mu_w) * grad


    def exact(self, t):
        Swi0 = self.s_0
        self.mu_water0 = self.mu_o*1.1
        def f(sat):
            return k_water(sat)/(k_water(sat)+self.mu_water0/self.mu_o*k_oil(sat))

        def df(sat):
            return (dk_water(sat)*(k_water(sat)+self.mu_water0/self.mu_o*k_oil(sat))-k_water(sat)*(dk_water(sat)+self.mu_water0/self.mu_o*dk_oil(sat)))/(k_water(sat)+self.mu_water0/self.mu_o*k_oil(sat))**2

        def opt(sat):
            return f(sat)-f(Swi0)-df(sat)*(sat-Swi0)

        Sc = fsolve(opt, 0.5)[0]
        Xc = self.Q/self.m0*(f(Sc)-f(Swi0))/(Sc-Swi0)*t
        Sx1 = np.zeros(200)
        X1 = np.zeros(200)
        Sx2 = np.zeros(200)
        X2 = np.zeros(200)
        Sx1[0] = 1
        Sx2[0] = Swi0
        X2[0] = Xc
        for i in range(1,200):
            Sx1[i] = Sx1[i-1]-(1-Sc)/199 
            Sx2[i] = Swi0
            X2[i] = X2[i-1] + (self.L+1e-3-Xc)/199
        for i in range(1,200):
            X1[i] = self.Q/self.m0 * df(Sx1[i]) * t
        X3 = np.array([Xc, Xc])
        Sx3 = np.array([Sc, Swi0])
        return np.concatenate((X1,X3,X2)), np.concatenate((Sx1,Sx3,Sx2))


    def solve(self):
        p, s = self.solution_init()
        s_buckley = np.zeros_like(s)
        s_buckley[0,:] = s[0,:]
        mu_oil_init = np.zeros(self.nx, dtype=np.float64)+self.mu_h
        mu_water_init = np.zeros(self.nx, dtype=np.float64)+self.mu_h*1.5
        self.mu_oil_arr.append(mu_oil_init)
        self.mu_water_arr.append(mu_water_init)
        self.grad_p.append(self.grad(p[0,:], self.dx))
        self.grad_mu_o.append(self.grad(p[0,:], self.dx))
        self.grad_mu_w.append(self.grad(p[0,:], self.dx))
        # Решение системы с использованием Ньютона-Крылова
        for t, t_step in tqdm(enumerate(self.time[:-1]), total=self.time.shape[0], desc="Processing"):
            # решение уравнения для давления
            p_old = p[t, :].copy()
            s_old = s[t, :].copy()
            p_new = self.solve_nonlinear(p_old, s_old, t_step, self.dt, self.dx)
            p[t+1, :] = p_new.copy()
            grad_new = np.gradient(p_new, self.dx)
            grad_old = np.gradient(p_old, self.dx)
            # решение уравнения для насыщенности
            fw_val = self.fw(s_old, p_new, grad_new, grad_old, self.mu_stop_w)
            s_new = np.zeros_like(s[t,:], dtype=np.float64)
            s_new[1:-1] = s_old[1:-1] + self.dt*self.p_0*self.T/(self.rho_w(p_new[1:-1])*self.m(p_new[1:-1])*self.dx*self.L**2) * (fw_val[1:-1] - fw_val[:-2]) - s_old[1:-1]*self.p_0*(self.beta_r/self.m(p_new[1:-1]) + self.beta_w*self.rho_w0/self.rho_w(p_new[1:-1]))*(p_new[1:-1]-p_old[1:-1])
            # s_new[1:-1] = np.where(s_new[1:-1]>1, 1, s_new[1:-1])
            # s_new[1:-1] = np.where(s_new[1:-1]<0, 0, s_new[1:-1])
            # s_new[0] = self.s_function(t_step)
            # s_new[0] = s_new[1]
            s_new[-1] = s_new[-2]
            s[t+1,1:] = s_new[1:].copy()

            mu_oil_new, res_o = self.mu_oil(grad_new, self.grad_mu_o[-1], self.mu_stop_o)
            mu_water_new, res_w = self.mu_water(grad_new, self.grad_mu_w[-1], self.mu_stop_w)
            self.mu_stop_o = res_o
            self.mu_stop_w = res_w
            self.grad_mu_o.append(res_o*self.L/self.p_0)
            self.grad_mu_w.append(res_w*self.L/self.p_0)
            self.grad_p.append(grad_new)
            self.mu_oil_arr.append(mu_oil_new)
            self.mu_water_arr.append(mu_water_new)
            x_b, s_b = self.exact(t_step*self.T)
            x_b = x_b/self.L
            func = interp1d(x_b, s_b, kind='nearest')
            s_b = func(self.x)
            s_buckley[t+1, :] = s_b
        self.mu_oil_arr = np.array(self.mu_oil_arr)
        self.mu_water_arr = np.array(self.mu_water_arr)
        self.grad_p = np.array(self.grad_p)
        self.grad_mu_o = np.array(self.grad_mu_o)
        self.grad_mu_w = np.array(self.grad_mu_w)

        self.p = p
        self.s = s
        return p, s, x_b, s_buckley