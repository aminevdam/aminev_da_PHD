from mpi4py import MPI
from dolfinx import mesh, fem, default_real_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from ufl import TrialFunction, TestFunction, TestFunctions, inner, grad, dx, dot, split
from basix.ufl import element, mixed_element
import numpy as np

# # Создание сетки (1D-сегмент от 0 до 1)
# msh = mesh.create_interval(MPI.COMM_WORLD, 100, [0.0, 1.0])

# # Определение функциональных пространств для насыщенности и давления
# P_space = fem.functionspace(msh, ("Lagrange", 1))
# S_space = fem.functionspace(msh, ("Lagrange", 1))

# # Создание смешанного функционального пространства
# mix = mixed_element([P_space.ufl_element(), S_space.ufl_element(), S_space.ufl_element()])
# W = fem.functionspace(msh, mix)

# # Определение пробных и тестовых функций для смешанного пространства
# u = fem.Function(W)  # current solution
# u0 = fem.Function(W)  # solution from previous converged step

# # Split mixed functions
# p, Sw, So = split(u)

# vp, vSw, vSo = TestFunctions(W)

# # Задаем начальные условия
# p_n, Sw_n, So_n = u0.split()

# # Начальные значения
# p_n.interpolate(lambda x: np.full_like(x[0], 1.0))  # Давление
# Sw_n.interpolate(lambda x: np.full_like(x[0], 0.2))  # Насыщенность водой
# So_n.interpolate(lambda x: np.full_like(x[0], 0.8))  # Насыщенность нефтью

# # Параметры задачи
# phi = 0.2  # Пористость
# k = 1e-12  # Абсолютная проницаемость (м^2)
# krw = lambda Sw: Sw**3  # Относительная проницаемость водной фазы
# kro = lambda So: So**2  # Относительная проницаемость нефтяной фазы
# mu_w = 1e-3  # Вязкость воды (Па·с)
# mu_o = 5e-3  # Вязкость нефти (Па·с)
# dt = 0.01  # Шаг по времени
# t_end = 1.0  # Конечное время
# num_steps = int(t_end / dt)  # Количество шагов по времени

# # Закон Дарси для водной и нефтяной фаз
# vw = -k * krw(Sw_n) / mu_w * grad(p_n)
# vo = -k * kro(So_n) / mu_o * grad(p_n)

# # Слабая форма для давления
# a_p = inner(grad(p), grad(vp)) * dx
# L_p = inner(vw + vo, grad(vp)) * dx

# # Слабая форма для насыщенности водной фазы
# a_Sw = phi * Sw * vSw * dx + dt * inner(grad(Sw), vw) * dx
# L_Sw = phi * Sw_n * vSw * dx

# # Слабая форма для насыщенности нефтяной фазы
# a_So = phi * So * vSo * dx + dt * inner(grad(So), vo) * dx
# L_So = phi * So_n * vSo * dx

# # Сборка матриц системы и создание решателя
# a = a_p + a_Sw + a_So
# L = L_p + L_Sw + L_So

# problem = LinearProblem(a, L)

# # Временная петля
# t = 0.0
# for i in range(num_steps):
#     t += dt
    
#     # Решение системы
#     w_h = problem.solve()
    
#     # Обновление значений для следующего шага
#     u0.x.array[:] = w_h.x.array

# # Вывод результатов
# p_h, Sw_h, So_h = w_h.split()
# print("Конечное давление:", p_h.x.array)
# print("Конечная насыщенность водой:", Sw_h.x.array)
# print("Конечная насыщенность нефтью:", So_h.x.array)




# Создание сетки (1D-сегмент от 0 до 1)
msh = mesh.create_interval(MPI.COMM_WORLD, 100, [0.0, 1.0])

# Определение функциональных пространств для насыщенности и давления
P_space = element("Lagrange", msh.basix_cell(), 1)
S_space = element("Lagrange", msh.basix_cell(), 1)

# Создание смешанного функционального пространства
mix = mixed_element([P_space, S_space, S_space])
W = fem.functionspace(msh, mix)

# Определение пробных и тестовых функций для смешанного пространства
vp, vSw, vSo = TestFunctions(W)

wh = fem.Function(W)
wn = fem.Function(W)

# Разделение функций на компоненты
p, Sw, So = split(wh)
p_n, Sw_n, So_n = split(wn)

# Zero u
wn.x.array[:] = 0.0

# Interpolate initial condition
wn.sub(0).interpolate(lambda x: np.full_like(x[0], 1.0))
wn.sub(1).interpolate(lambda x: np.full_like(x[0], 0.0))
wn.sub(2).interpolate(lambda x: np.full_like(x[0], 1.))

wn.x.scatter_forward()


# wh.x.array[:] = 0.0

# wh.sub(0).interpolate(lambda x: np.full_like(x[0], 1.0))
# wh.sub(1).interpolate(lambda x: np.full_like(x[0], 0.2))
# wh.sub(2).interpolate(lambda x: np.full_like(x[0], 0.8))

# wh.x.scatter_forward()

# Параметры задачи
phi = 0.2  # Пористость
k = 1e-12  # Абсолютная проницаемость (м^2)
krw = lambda Sw: Sw**2  # Относительная проницаемость водной фазы
kro = lambda So: 1 - Sw**2  # Относительная проницаемость нефтяной фазы
mu_w = 1e-3  # Вязкость воды (Па·с)
mu_o = 5e-3  # Вязкость нефти (Па·с)
dt = 0.01  # Шаг по времени
t_end = 1.0  # Конечное время
num_steps = int(t_end / dt)  # Количество шагов по времени

# Закон Дарси для водной и нефтяной фаз
vw = -k * krw(Sw) / mu_w * grad(p)
vo = -k * kro(So) / mu_o * grad(p)

# Слабая форма для давления
F_p = inner(vo, grad(vp)) * dx + inner(vw, grad(vp)) * dx

# Слабая форма для насыщенности водной фазы
F_Sw = phi * Sw * vSw * dx - phi * Sw_n * vSw * dx - dt * inner(vw, grad(vSw)) * dx 

# Слабая форма для насыщенности нефтяной фазы
F_So = phi * So * vSo * dx - dt * inner(vo, grad(vSo)) * dx - phi * So_n * vSo * dx

# Сборка матриц системы и создание решателя
F = F_p + F_Sw + F_So

problem = NonlinearProblem(F, wh)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "residual"
solver.rtol = 1e-6
solver.max_it = 100

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
ksp.setFromOptions()

# Временная петля
t = 0.0
for i in range(num_steps):
    t += dt
    
    # Решение системы
    solver.solve(wh)
    # Обновление значений для следующего шага
    wn.x.array[:] = wh.x.array

# Вывод результатов
p_h, Sw_h, So_h = wh.split()

print("Конечное давление:", p_h.x.array)
print("Конечная насыщенность водой:", Sw_h.x.array)
print("Конечная насыщенность нефтью:", So_h.x.array)

