"""Propogating Two Body Problem.

Imports twobodyproblem_cartesian ode from the ode_solvers module
for use with the solve_ivp integrator from scipy.

* 'TwoProblemPropogator' class for a single trajectory.

* 'MonteCarloPropogator' class for sampling from an initial
probability distribution and calculating the first 2 statistical moments

* 'keplerian_ics_doe' function constructs a random latin-hypercube
based on the input bounds dictionary (keys are the orbital elements and
the values are the corresponding bounds)

* 'main' shows an example run used for our application.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from pyDOE import lhs

import celestial_bodies_data as cb_data
from ode_solvers import twobodyproblem_cartesian as cart_ode
from ode_solvers import twobodyproblem_polar as polar_ode

from utils import cartesian_to_keplerian, \
                                 keplerian_to_cartesian, \
                                 build_covariance_matrix

plt.rcParams['text.usetex'] = True


class TwoBodyProblemPropogator:
    """class for orbit propogation of two body problem trajectory"""

    def __init__(self, state0, t_span,
                 central_body='earth', ode='cart_ode'):
        if ode == 'cart_ode':
            self.ode = cart_ode
            self.compute_elems = True
            self.orbitelems0 = cartesian_to_keplerian(self.states)[0]
            self.orbitelems = np.expand_dims(self.orbitelems0, 0)
            self.orbitelems_names = ['a', 'e', 'i', 'Omega', 'argp', 'nu']

        elif ode == 'polar_ode':
            self.ode = polar_ode
            self.compute_elems = False
        
        else:
            raise ValueError(f"ODE '{ode}' not known")
        
        self.state0 = state0
        self.states = np.expand_dims(state0, 0)

        self.t_span = t_span  # [s]
        self.epochs = [self.t_span[0]]

        self.central_body = central_body.lower()

    def propogate(self, t_eval=None, method='RK45', max_step=np.inf, tol=1e-6):
        """intgrate deterministic dynamics 'ode'"""
        sol = solve_ivp(
            fun=self.ode,
            t_span=self.t_span,
            y0=self.state0,
            method=method,
            t_eval=t_eval, max_step=max_step,
            rtol=tol, atol=tol,
            args=(
                cb_data.mu_bodies[self.central_body],
                )
        )
        if sol.status == -1:
            raise RuntimeError(sol.message)

        self.states = sol.y.transpose()
        if self.compute_elems:
            self.orbitelems = cartesian_to_keplerian(self.states)

        self.epochs = sol.t
        return self.states


class MonteCarloPropogator():
    """class for the Monte Carlo simulation"""
    def __init__(self, state0, t_span, n_samples=50,
                 initial_pdf_type='normal', initial_noise_std=None,
                 central_body='earth',):
        self.state0 = state0  # [km, km, km, km/s, km/s, km/s]
        self.orbitelems0 = cartesian_to_keplerian(self.state0)[0]
        self.orbitelems_names = ['a', 'e', 'i', 'Omega', 'argp', 'nu']

        self.t_span = t_span  # [s]
        self.epochs = [self.t_span[0]]

        self.central_body = central_body.lower()

        self.n_samples = n_samples
        self.initial_pdf_type = initial_pdf_type

        if initial_noise_std is None:
            self.initial_noise_std = dict(
                zip(self.orbitelems_names, [
                    30e-3, 30e-3, 36*4.84814e-6,
                    36*4.84814e-6, 36*4.84814e-6, 36*4.84814e-6
                    ]))
        else:
            self.initial_noise_std = initial_noise_std

        self.pdf_orbitelems0 = self.__initial_probability_distributions(
            self.n_samples, self.initial_noise_std, self.initial_pdf_type)

        self.pdf_state0 = keplerian_to_cartesian(self.pdf_orbitelems0)

        self.state_flows = np.expand_dims(self.pdf_state0, 1)
        self.orbitelems_flows = np.expand_dims(self.pdf_orbitelems0, 1)
        self.means = np.mean(self.state_flows, 0)
        self.means_orbitelems = np.mean(self.orbitelems_flows, 0)
        self.covs = build_covariance_matrix(self.state_flows)
        self.covs_orbitelems = build_covariance_matrix(self.orbitelems_flows)

    def __initial_probability_distributions(
            self, n_samples, initial_noise_std, initial_pdf_type):
        pdf_orbitelems0 = []
        for (element_name, element_value) in zip(self.orbitelems_names,
                                                 self.orbitelems0):
            if initial_pdf_type.lower() == 'normal':
                self.initial_pdf_type = initial_pdf_type
                pdf_orbitelems0.append(
                    np.random.normal(loc=element_value,
                                     scale=initial_noise_std[element_name],
                                     size=(n_samples)))
            else:
                raise(ValueError(
                    f'distribution {initial_pdf_type} not known by module.'))

        pdf_orbitelems0 = np.asarray(pdf_orbitelems0).transpose()
        return pdf_orbitelems0

    def propogate(
            self, t_eval=None, method='RK45', max_step=np.inf, tol=1e-7):
        """propogate samples of the initial probability distribution"""
        state_flows = []
        orbitelems_flows = []
        for state0 in self.pdf_state0:
            tbp_obj = TwoBodyProblemPropogator(
                state0, self.t_span, self.central_body)
            tbp_obj.propogate(t_eval=t_eval, method=method,
                              max_step=max_step, tol=tol)
            state_flows.append(tbp_obj.states)
            orbitelems_flows.append(tbp_obj.orbitelems)

        # set up interpolation for algorithm chosen time steps ???
        if t_eval is None:
            self.state_flows = state_flows
            self.orbitelems_flows = orbitelems_flows
            self.means = None
            self.covs = None
        else:
            self.epochs = t_eval
            self.state_flows = np.asarray(state_flows)
            self.orbitelems_flows = np.asarray(orbitelems_flows)
            self.means = np.mean(self.state_flows, 0)
            self.means_orbitelems = np.mean(self.orbitelems_flows, 0)
            self.covs = build_covariance_matrix(self.state_flows)
            self.covs_orbitelems = build_covariance_matrix(
                self.orbitelems_flows)
        return self.means, self.covs


def keplerian_ics_doe(
        n_ics, keplerian_bounds, coords='cart', central_body='earth'):
    """Generate n_ics initial conditions of classical orbital elemennts"""

    if n_ics == 1:
        keplerian_doe = np.random.uniform(size=(1, 6)) * (
            np.max(keplerian_bounds, 1) - np.min(keplerian_bounds, 1)) + \
            np.min(keplerian_bounds, 1)
    else:
        lhc_doe = lhs(6, n_ics, 'm', 20)
        keplerian_doe = lhc_doe * (
            np.max(keplerian_bounds, 1) - np.min(keplerian_bounds, 1)) + \
            np.min(keplerian_bounds, 1)

    # CONVERT TO CARTESIAN COORDS
    ics_cart = keplerian_to_cartesian(keplerian_doe, central_body)
    if coords == 'cart':
        return ics_cart, keplerian_doe
    elif coords == 'polar':
        ics_polar = []
        for state0_cart in ics_cart:
            [x, y, z, vx, vy, vz] = state0_cart
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan(x / y)  # first quadrant
            if (x < 0) and (y > 0):
                theta = theta - np.pi  # second quadrant
            if (x < 0) and (y < 0):
                theta = theta + np.pi  # third quadrant
            if (x > 0) and (y < 0):
                theta = 2 * np.pi + theta  # fourth quadrant

            vr = (x * vx + y * vy) / np.sqrt(x**2 + y**2)
            vt = r * (x * vy - y * vx) / (x**2 + y**2)
            state0_polar = np.array([r, theta, vr, vt])
            ics_polar.append(state0_polar)
        ics_polar = np.asarray(ics_polar)
        return ics_polar, keplerian_doe


if __name__ == "__main__":
    from time import time

    # NOMINAL TRAJECTORIES DOE
    N_ICS = 1000

    a_bounds = [1000, 2000]  # semi-major axis [km]
    e_bounds = [0, 0.5]  # eccentricity [-]
    i_bounds = [0, np.pi]  # inclination [rad]
    Omega_bounds = [0, 2*np.pi]  # right ascension [rad]
    argp_bounds = [0, 2*np.pi]  # argument of periapsis [rad]
    nu_bounds = [0, 2*np.pi]  # true anomaly [rad]

    keplerian_bounds = np.asarray([
            a_bounds, e_bounds, i_bounds,
            Omega_bounds, argp_bounds, nu_bounds
        ]
    )

    cart_ics, kepler_ics = keplerian_ics_doe(N_ICS, keplerian_bounds)

    # INITIAL PDF
    range_err = 30e-3  # [km]
    angle_err = 36*4.84814e-6  # [rad]

    noise_std = {
        'a': range_err,
        'e': range_err,
        'i': angle_err,
        'Omega': angle_err,
        'argp': angle_err,
        'nu': angle_err
    }

    distribution_type = 'normal'

    # PROPOGATE SAMPLES OF EACH INITIAL PDF
    T0 = 0
    TF = 3600
    DT = 50
    epochs = np.arange(T0, TF, DT)

    m_samples = 500  # number of samples

    monte_carlo_sims = []
    tic = time()
    tic = time()
    for i, state0_nominal in enumerate(cart_ics):
        mc_sim = MonteCarloPropogator(
            state0=state0_nominal,
            t_span=[T0, TF],
            n_samples=m_samples,
            initial_pdf_type=distribution_type,
            initial_noise_std=noise_std,
            central_body='earth'
        )
        mc_sim.propogate(
            t_eval=epochs,
            method='RK45',
            max_step=DT,
            tol=1e-10
        )
        monte_carlo_sims.append(mc_sim)

        # progress bar for output
        eta = round((time()-tic) * (N_ICS/(i+1) - 1)/60, 2)
        print(
            f'\rCompleted Trajectory : {i+1}/{N_ICS}\tETA [mins] : {eta}',
            end='\r')
    print(f'SIMULATION COMPLETE')

    monte_carlo_sims = np.array(monte_carlo_sims, dtype=object)

    # save .npy array of <MonteCarloPropogator> objects
    file_prefix = 'data'
    file_path = f'{file_prefix}\\monte_carlo_sims_{N_ICS}' \
                f'ics_{m_samples}smp.npy'
    np.save(file_path, monte_carlo_sims)
