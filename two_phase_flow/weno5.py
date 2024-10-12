import numpy as np

def weno5_flux(Sw):
    flux = np.zeros_like(Sw)
    eps = 1e-6  # Для предотвращения деления на 0
    flux[0] = (2*Sw[0] + 5*Sw[1] - Sw[2]) / 6
    flux[1] = (-Sw[0] + 5*Sw[1] + 2*Sw[2]) / 6
    flux[-1] = (2*Sw[-3] - 7*Sw[-2] + 11*Sw[-1]) / 6
    flux[-2] = (-Sw[-3] + 5*Sw[-2] + 2*Sw[-1]) / 6

    for i in range(2, len(Sw) - 2):
        # Левые потоки
        f1 = (2*Sw[i-2] - 7*Sw[i-1] + 11*Sw[i]) / 6
        f2 = (-Sw[i-1] + 5*Sw[i] + 2*Sw[i+1]) / 6
        f3 = (2*Sw[i] + 5*Sw[i+1] - Sw[i+2]) / 6

        # Веса для WENO
        beta1 = 13/12 * (Sw[i-2] - 2*Sw[i-1] + Sw[i])**2 + 1/4 * (Sw[i-2] - 4*Sw[i-1] + 3*Sw[i])**2
        beta2 = 13/12 * (Sw[i-1] - 2*Sw[i] + Sw[i+1])**2 + 1/4 * (Sw[i-1] - Sw[i+1])**2
        beta3 = 13/12 * (Sw[i] - 2*Sw[i+1] + Sw[i+2])**2 + 1/4 * (3*Sw[i] - 4*Sw[i+1] + Sw[i+2])**2

        alpha1 = 0.1 / (eps + beta1)**2
        alpha2 = 0.6 / (eps + beta2)**2
        alpha3 = 0.3 / (eps + beta3)**2

        w1 = alpha1 / (alpha1 + alpha2 + alpha3)
        w2 = alpha2 / (alpha1 + alpha2 + alpha3)
        w3 = alpha3 / (alpha1 + alpha2 + alpha3)

        flux[i] = w1 * f1 + w2 * f2 + w3 * f3
    
    return flux
