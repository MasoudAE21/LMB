import matplotlib.pyplot as plt
import numpy as np


def plot_velocity(ux, uy):

    speed = np.sqrt(ux**2 + uy**2)

    plt.figure(figsize=(6,6))

    plt.imshow(speed,
               origin="lower",
               cmap="jet")

    plt.colorbar(label="Velocity")

    plt.streamplot(
        np.arange(speed.shape[1]),
        np.arange(speed.shape[0]),
        ux,
        uy,
        color="white",
        density=1.5
    )

    plt.tight_layout()

    plt.show()