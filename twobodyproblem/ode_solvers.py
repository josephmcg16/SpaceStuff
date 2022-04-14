"""ode solvers for space stuff!"""
import numpy as np


def twobodyproblem_cartesian(epoch, state_vec, mu_body):
    """ode for the two_body problem
    extend for additional forcing models?
    variation in mu_body?"""
    pos_vec = state_vec[:3]
    vel_vec = state_vec[3:]
    acc_vec = -mu_body * (pos_vec/(np.linalg.norm(pos_vec) ** 3))
    return np.hstack([vel_vec, acc_vec])
