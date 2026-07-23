import numpy as np

from config import *

from solver.initialize import initialize
from solver.macroscopic import macroscopic
from solver.collision import collide
from solver.streaming import stream
from boundary.bounce_back import apply_bounce_back
from boundary.moving_wall import moving_lid
from utils.plotting import plot_velocity


f, rho, ux, uy = initialize(
    NX,
    NY,
    rho0,
    ux0,
    uy0
)

for step in range(num_steps):

    rho, ux, uy = macroscopic(f)
    print(f"Step: {step}, rho: {rho.mean()}, ux: {ux.mean()}, uy: {uy.mean()}")
    f = collide(
        f,
        rho,
        ux,
        uy,
        tau
    )

    f = stream(f)
    f = apply_bounce_back(f)
    f = moving_lid(f, rho, 0.02)

plot_velocity(ux, uy)
    

print("Simulation finished.")