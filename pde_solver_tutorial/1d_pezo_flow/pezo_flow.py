# решение уравнения пьезопроводности

import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot, default_scalar_type
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from ufl import TrialFunction, TestFunction, inner, dx, grad, dot
import matplotlib.pyplot as plt
from copy import deepcopy

# Define mesh
nx = 50
domain = mesh.create_interval(MPI.COMM_WORLD, nx, points=np.array([0, 1]))
V = fem.functionspace(domain, ("Lagrange", 1))

u_n = fem.Function(V)
u_n.name = "u_n"

uh = fem.Function(V)
uh.name = "u_h"

#начальное условие
u_n.interpolate(lambda x: np.full_like(x[0], 1.))
uh.interpolate(lambda x: np.full_like(x[0], 1.))

boundary_dofs_0 = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.))
bc_0 = fem.dirichletbc(ScalarType(0.5), boundary_dofs_0, V)

boundary_dofs_1 = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1.))
bc_1 = fem.dirichletbc(ScalarType(1.), boundary_dofs_1, V)

bc = [bc_0, bc_1]

alpha = 0.1   # Коэффициент теплопроводности
dt = 0.001      # Шаг по времени
t_end = 1.0    # Конечное время
num_steps = int(t_end / dt)  # Количество шагов по времени

# Определяем вариационные формы
u = TrialFunction(V)
v = TestFunction(V)

# Левая часть уравнения (используем метод Галёркина)
a = u * v * dx + dt * alpha * dot(grad(u), grad(v)) * dx

# Правая часть уравнения (с использованием предыдущего решения u_n)
L = (u_n * v * dx)

bilinear_form = fem.form(a)
linear_form = fem.form(L)

# Сборка матрицы системы
A = assemble_matrix(bilinear_form, bcs=bc)
A.assemble()
b = create_vector(linear_form)


# Настройка решателя
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType("preonly")
solver.getPC().setType("lu")

# Временная петля
t = 0.0

sol = [deepcopy(u_n.x.array)]

for i in range(num_steps):
    t += dt
    
    # Сборка вектора правой части
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)
    
    # Применение граничных условий
    fem.petsc.apply_lifting(b, [bilinear_form], bcs=[[bc_0, bc_1]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, [bc_0, bc_1])
    
    # Решение системы
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()
    
    # Обновление решения u_n
    u_n.x.array[:] = uh.x.array

    sol.append(deepcopy(u_n.x.array))

# Вывод решения в конечный момент времени
print("Решение в конечный момент времени t =", t)

for i in sol:
    plt.plot(np.linspace(0,1,51), i)
plt.show()