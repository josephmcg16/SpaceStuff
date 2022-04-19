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


def twobodyproblem_polar(epoch, state_vec, mu_body):
    """ODE for 2-body problem in polar coordinates.

    :param t: time
    :param state_vec: state vector, [r, theta, v_r, v_theta]
    :param mu_body: gravitational parameter
    :return: state derivative wrt time
    """
    pos_vec = state_vec[:2]
    vel_vec = state_vec[2:]
    acc_vec = [-mu_body / pos_vec[0]**2 + vel_vec[1] ** 2 / pos_vec[0],
               -(vel_vec[0]*vel_vec[1])/pos_vec[0]]
    return np.hstack([vel_vec[0], vel_vec[1] / pos_vec[0], acc_vec])
