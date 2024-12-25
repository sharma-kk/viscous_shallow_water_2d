import os
os.environ["OMP_NUM_THREADS"] = "1"
# import numpy as np
from firedrake import *
import math
import time
from firedrake.petsc import PETSc

Nx = 448
Ny = 128 # resolution 60 km
mesh = PeriodicRectangleMesh(Nx, Ny, 3.5, 1, direction="x")

V1 = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "DG", 0)

x, y = SpatialCoordinate(mesh)

# define dimensionless parameters
Ro = 0.4 ; Re = 6*10**3 ; B = 6.93 ; C = 0.06 ; Pe = 6*10**3

# define initial condtions
y0 = 11/49 ; y1 = 38/49
alpha = 0.82
u0_1 = conditional(Or(y <= y0, y >= y1), 0.0, exp(alpha**2/((y - y0)*(y - y1)))*exp(4*alpha**2/(y1 - y0)**2))
u0_2 = 0.0

u0 = project(as_vector([u0_1, u0_2]), V1)
g = project(as_vector([u0_2, -(C/Ro)*(1 + B*y)*u0_1]), V1)

f = interpolate(div(g), V0)

h0 = TrialFunction(V2)
q = TestFunction(V2)

a = -inner(grad(h0), grad(q))*dx
L = f*q*dx

h0 = Function(V2) # geostrophic height
nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD) # this is required with Neumann bcs
solve(a == L, h0, nullspace=nullspace)

# height perturbation
h0_c = 1.0
c0 = 0.01 ; c1 = 16 ;  c2 = 324 ; x_0 = 1.75; y_2 = 0.5
h0_p = interpolate(c0*cos(math.pi*y/2)*exp(-c1*(x  - x_0)**2)*exp(-c2*(y - y_2)**2), V2)

h0_f = interpolate(h0_c + h0 + h0_p, V2) # perturbed initial height

# Variational formulation
Z = V1*V2

uh = Function(Z)
u, h = split(uh)
v, phi = TestFunctions(Z)
u_ = Function(V1)
h_ = Function(V2)

u_.assign(u0)
h_.assign(h0_f)

perp = lambda arg: as_vector((-arg[1], arg[0]))

Dt =0.01 # roughly 16 minutes

F = ( inner(u-u_,v)
    + Dt*0.5*(inner(dot(u, nabla_grad(u)), v) + inner(dot(u_, nabla_grad(u_)), v))
    + Dt*0.5*(1/Ro)*inner((1 + B*y)*(perp(u) + perp(u_)), v)
    - Dt*0.5*(1/C)*(h + h_)* div(v)
    + Dt *0.5 *(1/Re)*inner((nabla_grad(u)+nabla_grad(u_)), nabla_grad(v))
    + (h - h_)*phi - Dt*0.5*inner(h_*u_ + h*u, grad(phi))
    + Dt*0.5*(1/Pe)*inner((grad(h)+grad(h_)),grad(phi)) )*dx

bound_cond = [DirichletBC(Z.sub(0).sub(1), Constant(0.0), (1,2))]

# visulization at t=0
h_.rename("height")
u_.rename("velocity")

outfile = File("./results/baro_inst_mesh_128.pvd")
outfile.write(u_, h_)

# time stepping and visualization at other time steps
t_start = Dt
t_end = Dt*540 # 144 hours

t = Dt
iter_n = 1
freq = 90
# freq = 5
t_step = freq*Dt # 24 hours
current_time = time.strftime("%H:%M:%S", time.localtime())
PETSc.Sys.Print("Local time at the start of simulation:",current_time)
start_time = time.time()

while (round(t,4) <= t_end):
    solve(F == 0, uh, bcs = bound_cond)
    u, h = uh.subfunctions
    if iter_n%freq == 0:
        if iter_n == freq:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 # running time for one time step (t_step)
            PETSc.Sys.Print("Approx. running time for one t_step: %.2f minutes" %execution_time)
            total_execution_time = ((t_end - t_start)/t_step)*execution_time
            PETSc.Sys.Print("Approx. total running time: %.2f minutes:" %total_execution_time)

        PETSc.Sys.Print("t=", round(t,4))
        h.rename("height")
        u.rename("velocity")
        outfile.write(u, h)
    u_.assign(u)
    h_.assign(h)

    t += Dt
    iter_n +=1
