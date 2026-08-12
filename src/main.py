# import numpy as np

# from config import *

# from lbm.collision import collide
# from lbm.streaming import stream
# from state import LBMState
# from utils.plotting import plot_velocity


# state = LBMState(NX, NY)

# state.initialize_flow()

# for _ in range(num_steps):

#     state.save_previous_state()

#     state.compute_flow()

#     f_intermediate = collide(
#         state.f,
#         state.rho,
#         state.ux,
#         state.uy,
#         tau
#     )

#     f_intermediate = stream(f_intermediate)

#     # f_intermediate = boundary_manager.apply_flow(f_intermediate)
    
#     state.update_flow_distribution(f_intermediate)

#     state.next_iteration()

#     if state.residual() < 1e-8:
#         print("Converged")
#         break

# plot_velocity(state.ux, state.uy)
    

# print("Simulation finished.")

from problems.problem_2_transient_conduction import run as problem_2
from problems.problem_1_natural_convection import run as problem_1
from utils.plotting import plot_results

# problem_2(thermal_diffusivity=1.0)
result = problem_1(Ra=1e7 ,visco=0.2, nx=550, ny=550, max_steps=150000)
if result is not None:
    plot_results(result)

# result = problem_1(
#     Ra=1e3,
#     nx=101,
#     ny=101,
#     max_steps=3000
# )

# plot_results(result)