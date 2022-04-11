"""misc functions used by other modules.
This is not meant to be tidy... sorry :("""
import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
from celestial_bodies_data import mu_bodies


def cartesian_to_keplerian(states, central_body='earth', tol=1e-10):
    """from 'Fundamentals of Astrodynamics and Applications'
    - Vallado (2007)"""
    mu = mu_bodies[central_body.lower()]

    if np.asarray(states).ndim == 1:
        states = np.expand_dims(states, 0)

    a_list = []
    e_list = []
    i_list = []
    Omega_list = []
    argp_list = []
    nu_list = []
    for state in states:  # loop over each time-series state observation
        r_vec = state[:3]
        v_vec = state[3:]

        h_vec = np.cross(r_vec, v_vec)

        n_vec = np.cross([0, 0, 1], h_vec)

        e_vec = (1/mu) * (
            (norm(v_vec)**2 - mu/norm(r_vec)) * r_vec -
            np.dot(r_vec, v_vec) * v_vec
        )
        e = norm(e_vec)
        E = norm(v_vec)**2 / 2 - mu/norm(r_vec)

        if abs(e - 1) > tol:
            a = - mu / (2 * E)
        else:
            a = np.inf

        i = np.arccos(h_vec[2]/norm(h_vec))

        Omega = np.arccos(n_vec[0]/norm(n_vec))

        argp = np.arccos(np.dot(n_vec, e_vec)/(norm(n_vec)*e))

        nu = np.arccos(np.dot(e_vec, r_vec)/(e*norm(r_vec)))

        if n_vec[1] < 0:
            Omega = 2*np.pi - Omega

        if e_vec[2] < 0:
            argp = 2*np.pi - argp

        if np.dot(r_vec, v_vec) < 0:
            nu = 2*np.pi - nu

        a_list.append(a)
        e_list.append(e)
        i_list.append(i)
        Omega_list.append(Omega)
        argp_list.append(argp)
        nu_list.append(nu)

    a = np.vstack(a_list)
    e = np.vstack(e_list)
    i = np.vstack(i_list)
    Omega = np.vstack(Omega_list)
    argp = np.vstack(argp_list)
    nu = np.vstack(nu_list)

    return np.hstack([a, e, i, Omega, argp, nu])


def keplerian_to_cartesian(keplerian_elements, central_body='earth'):
    """convert from keplerian elements to cartesian coordinates
    translated from MATHWORKS keplerian2ijk function from Aerospace Toolbox
    https://uk.mathworks.com/help/aerotbx/index.html?s_tid=CRUX_lftnav"""
    mu = mu_bodies[central_body.lower()]

    a = keplerian_elements[:, 0]  # semi-major axis [km]
    e = keplerian_elements[:, 1]  # eccentricity [-]
    i = keplerian_elements[:, 2]  # inclination [rad]
    Omega = keplerian_elements[:, 3]  # right ascension of ascending node [rad]
    argp = keplerian_elements[:, 4]  # argument of periapsis [rad]
    nu = keplerian_elements[:, 5]  # true anomaly [rad]

    peri = a*(1-e**2)
    r_0 = peri / (1 + e * np.cos(nu))

    x_pos = r_0 * np.cos(nu)
    y_pos = r_0 * np.sin(nu)

    vel_x_ = -(mu/peri) ** (1/2) * np.sin(nu)
    vel_y_ = (mu/peri) ** (1/2) * (e + np.cos(nu))

    x_coord = (np.cos(Omega) * np.cos(argp) - np.sin(Omega) * np.sin(argp) *
               np.cos(i)) * x_pos + \
        (-np.cos(Omega) * np.sin(argp) - np.sin(Omega) * np.cos(argp) *
         np.cos(i)) * y_pos

    y_coord = (np.sin(Omega) * np.cos(argp) + np.cos(Omega) * np.sin(argp) *
               np.cos(i)) * x_pos + \
        (-np.sin(Omega) * np.sin(argp) + np.cos(Omega) * np.cos(argp) *
         np.cos(i)) * y_pos

    z_coord = (np.sin(argp) * np.sin(i)) * x_pos + (np.cos(argp) *
                                                    np.sin(i)) * y_pos

    vel_x_coord = (np.cos(Omega) * np.cos(argp) -
                   np.sin(Omega) * np.sin(argp) * np.cos(i)) * vel_x_ + \
        (-np.cos(Omega) * np.sin(argp) - np.sin(Omega) * np.cos(argp) *
         np.cos(i)) * vel_y_

    vel_y_coord = (np.sin(Omega) * np.cos(argp) +
                   np.cos(Omega) * np.sin(argp) * np.cos(i)) * vel_x_ + \
        (-np.sin(Omega) * np.sin(argp) + np.cos(Omega) * np.cos(argp) *
         np.cos(i)) * vel_y_

    vel_z_coord = (np.sin(argp) * np.sin(i)) * vel_x_ + \
                  (np.cos(argp) * np.sin(i)) * vel_y_

    return np.vstack(
        [np.vstack([x_coord, y_coord, z_coord]),
         np.vstack([vel_x_coord, vel_y_coord, vel_z_coord])]).transpose()


def build_covariance_matrix(state_flows):
    """covariance of a state vector over multiple time steps"""
    p_tensor = []
    for pdf_state in state_flows.transpose(1, 2, 0):
        p_tensor.append(np.cov(pdf_state))
    return np.asarray(p_tensor)


def plot_trajectory(states):
    """3d plot of a single trajectory"""
    fig, axes = plt.subplots(subplot_kw={'projection': '3d'})
    axes.scatter(0, 0, 0)
    axes.plot(states[0, 0], states[0, 1], states[0, 2],
              marker='*', markersize=7)
    axes.plot(states[:, 0], states[:, 1], states[:, 2])
    axes.set_xlabel(r'$\mathbf{r}_1$ [km]')
    axes.set_ylabel(r'$\mathbf{r}_2$ [km]')
    axes.set_zlabel(r'$\mathbf{r}_3$ [km]')
    return fig, axes
