from problems.problem_2_transient_conduction import run as problem_2
from problems.problem_1_natural_convection import run as problem_1
from utils.plotting import plot_results

case_1 = True
if case_1:
    result = problem_1(Ra=1e5 ,visco=0.02, nx=100, ny=100, max_steps=150000, tolerance=1e-8)
    if result is not None:
        plot_results(result)
else:
    problem_2(thermal_diffusivity=1.0)

