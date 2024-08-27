from mpi4py import MPI
import numpy
import ufl
import pyvista

from dolfinx import default_scalar_type
from dolfinx import mesh, fem, plot
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem

# создание области для решения
domain = mesh.create_unit_square(MPI.COMM_WORLD, 20, 20, mesh.CellType.quadrilateral)

# определение функционального пространства на области решения
V = functionspace(domain, ("Lagrange", 2))

# определение функции, которая задает начальные условия
uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)


# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

# определение граничного условия дирихле
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

# определение целевой и тестовой функций
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# задание константной правой части
f = fem.Constant(domain, default_scalar_type(-6))

# определение вариационной задачи
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# определение решателя СЛУ
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# решение СЛУ
uh = problem.solve()

# определение ошибок вычисления
V2 = fem.functionspace(domain, ("Lagrange", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

error_max = numpy.max(numpy.abs(uD.x.array-uh.x.array))
# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")

# вывод графиков
try:
    import pyvista

    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("uh_poisson.png")
    else:
        plotter.show()
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
# from dolfinx import plot
# pyvista.start_xvfb()
# domain.topology.create_connectivity(tdim, tdim)
# topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True)
# plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     plotter.show()
# else:
#     figure = plotter.screenshot("fundamentals_mesh.png")

# u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
# u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# u_grid.point_data["u"] = uh.x.array.real
# u_grid.set_active_scalars("u")
# u_plotter = pyvista.Plotter()
# u_plotter.add_mesh(u_grid, show_edges=True)
# u_plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     u_plotter.show()