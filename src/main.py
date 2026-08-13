from problems.problem_2_transient_conduction import run as problem_2
from problems.problem_1_natural_convection import run as problem_1
from utils.plotting import plot_results

case_1 = True
if case_1:
    result = problem_1(Ra=1e7, nx=100, ny=600, max_steps=150000, tolerance=2e-5)
    if result is not None:
        plot_results(result)
else:
    problem_2(thermal_diffusivity=0.1, nx=50, max_steps=100000, tol=1e-6)

