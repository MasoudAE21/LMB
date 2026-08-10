# import numpy as np
# from copy import deepcopy

# from lattice.equilibrium import equilibrium
# from lattice.d2q9 import D2Q9


# class LBMState:
#     """
#     Stores the complete state of an LBM simulation.

#     Flow:
#         f, rho, ux, uy

#     Scalar:
#         g, C

#     Also optionally stores the previous timestep.
#     """

#     def __init__(self,
#                  nx: int,
#                  ny: int,
#                  save_previous: bool = True):

#         self.nx = nx
#         self.ny = ny

#         self.iteration = 0

#         self.save_previous = save_previous

#         # ---------- Flow ----------
#         self._f = None
#         self._rho = None
#         self._ux = None
#         self._uy = None

#         # ---------- Scalar ----------
#         self._g = None
#         self._C = None

#         # ---------- Previous ----------
#         self.previous = None

#     @property
#     def f(self):
#         return self._f
    
#     @property
#     def rho(self):
#         return self._rho
    
#     @property
#     def ux(self):
#         return self._ux
    
#     @property
#     def uy(self):
#         return self._uy
    
#     @property
#     def g(self):
#         return self._g
    
#     @property
#     def C(self):
#         return self._C

#     def update_flow_distribution(self, f):

#         if self.save_previous:
#             self.previous_f = None if self._f is None else self._f.copy()

#         self._f = f

#         self.compute_flow()

#     def update_scalar_distribution(self, g):

#         if self.save_previous:
#             self.previous_g = None if self._g is None else self._g.copy()

#         self._g = g

#         self.compute_scalar()
    
#     def initialize_flow(self,
#                         rho0=1.0,
#                         ux0=0.0,
#                         uy0=0.0):

#         self._rho = np.full((self.ny, self.nx), rho0)

#         self._ux = np.full((self.ny, self.nx), ux0)

#         self._uy = np.full((self.ny, self.nx), uy0)

#         self._f = equilibrium(
#             self._rho,
#             self._ux,
#             self._uy
#         )

#     def initialize_scalar(self,
#                           C0=0.0):

#         self._C = np.full(
#             (self.ny, self.nx),
#             C0
#         )

#         self._g = np.zeros((9, self.ny, self.nx))

#         for i in range(D2Q9.Q):
#             self._g[i] = D2Q9.w[i] * self._C

#     def compute_flow(self):

#         self._rho = np.sum(self._f, axis=0)

#         self._ux = np.zeros_like(self._rho)
#         self._uy = np.zeros_like(self._rho)

#         for i in range(D2Q9.Q):

#             self._ux += D2Q9.c[i, 0] * self._f[i]
#             self._uy += D2Q9.c[i, 1] * self._f[i]

#         self._ux /= self._rho
#         self._uy /= self._rho

#     def compute_scalar(self):

#         self._C = np.sum(self._g, axis=0)

#     def save_previous_state(self):

#         if not self.save_previous:
#             return

#         self.previous = {
#             "f": self.f.copy() if self.f is not None else None,
#             "g": self.g.copy() if self.g is not None else None,
#             "rho": self.rho.copy() if self.rho is not None else None,
#             "ux": self.ux.copy() if self.ux is not None else None,
#             "uy": self.uy.copy() if self.uy is not None else None,
#             "C": self.C.copy() if self.C is not None else None,
#         }

#     def residual(self):

#         if self.previous is None:
#             return np.inf

#         if self.f is None:
#             return np.inf

#         return np.max(
#             np.abs(
#                 self.f - self.previous["f"]
#             )
#         )

#     def next_iteration(self):

#         self.iteration += 1